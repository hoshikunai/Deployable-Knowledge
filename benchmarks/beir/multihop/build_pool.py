from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BEIR_ROOT = Path(__file__).resolve().parents[1]
if str(BEIR_ROOT) not in sys.path:
    sys.path.insert(0, str(BEIR_ROOT))

from dataset_catalog import DATASET_NAME_PATTERN  # noqa: E402

DATASETS_ROOT = BEIR_ROOT / "datasets"
BUILDER_VERSION = "multihop-pool-v1"
NAME_PREFIX = "multihop-"
SPLITS = ("dev", "test")
MIN_WORDS = 5
MAX_TITLE_WORDS = 12
ID_PREFIXES = {"hotpotqa": "hp", "musique": "mq", "2wiki": "tw"}


class SourceFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Paragraph:
    title: str
    text: str
    gold: bool
    hop: int | None


@dataclass(frozen=True)
class PoolQuestion:
    id: str
    question: str
    type: str
    level: str | None
    answers: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]
    supporting_facts: tuple[tuple[str, int], ...]
    decomposition: tuple[dict[str, Any], ...]
    triples: tuple[tuple[str, str, str], ...]

    @property
    def gold(self) -> list[Paragraph]:
        return [paragraph for paragraph in self.paragraphs if paragraph.gold]


@dataclass
class LoadResult:
    questions: list[PoolQuestion]
    dropped: Counter[str]


def require(record: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in record:
        raise SourceFormatError(f"{where}: missing '{key}'")
    value = record[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise SourceFormatError(f"{where}: '{key}' has type {type(value).__name__}")
    return value


def require_record(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourceFormatError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def title_variants(title: str) -> set[str]:
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    return {variant for variant in (normalize(title), normalize(stripped)) if variant}


def mentions_title(question: str, title: str) -> bool:
    padded = f" {normalize(question)} "
    return any(f" {variant} " in padded for variant in title_variants(title))


def word_count(title: str, text: str) -> int:
    return len("\n\n".join(value for value in (title.strip(), text.strip()) if value).split())


def join_sentences(sentences: list[Any], where: str) -> str:
    if not all(isinstance(sentence, str) for sentence in sentences):
        raise SourceFormatError(f"{where}: sentences must be strings")
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip())


def heuristic_hops(question: str, gold_titles: list[str]) -> dict[str, int | None]:
    named = {title for title in gold_titles if mentions_title(question, title)}
    if not named:
        return {title: None for title in gold_titles}
    return {title: 1 if title in named else 2 for title in gold_titles}


def load_wiki_style(records: Any, with_triples: bool) -> LoadResult:
    if not isinstance(records, list):
        raise SourceFormatError("source: expected a JSON array of questions")
    questions: list[PoolQuestion] = []
    dropped: Counter[str] = Counter()
    for index, raw in enumerate(records):
        where = f"record {index}"
        record = require_record(raw, where)
        question_id = require(record, "_id", str, where)
        where = f"question {question_id}"
        question = require(record, "question", str, where)
        context = require(record, "context", list, where)
        facts = require(record, "supporting_facts", list, where)
        answer = require(record, "answer", str, where)
        question_type = require(record, "type", str, where)
        level = require(record, "level", str, where) if "level" in record else None

        titles: list[str] = []
        texts: list[str] = []
        for position, entry in enumerate(context):
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) \
                    or not isinstance(entry[1], list):
                raise SourceFormatError(f"{where}: context[{position}] must be [title, [sentences]]")
            titles.append(entry[0])
            texts.append(join_sentences(entry[1], f"{where}: context[{position}]"))
        if len(set(titles)) != len(titles):
            dropped["duplicate-context-title"] += 1
            continue

        supporting: list[tuple[str, int]] = []
        for position, fact in enumerate(facts):
            if not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str) \
                    or not isinstance(fact[1], int):
                raise SourceFormatError(f"{where}: supporting_facts[{position}] must be [title, index]")
            if fact[0] not in titles:
                raise SourceFormatError(f"{where}: supporting title {fact[0]!r} is not in the context")
            supporting.append((fact[0], fact[1]))
        gold_titles = list(dict.fromkeys(title for title, _ in supporting))
        if not gold_titles:
            raise SourceFormatError(f"{where}: no supporting facts")

        triples: list[tuple[str, str, str]] = []
        if with_triples:
            for position, triple in enumerate(require(record, "evidences", list, where)):
                if not isinstance(triple, list) or len(triple) != 3 \
                        or not all(isinstance(part, str) for part in triple):
                    raise SourceFormatError(f"{where}: evidences[{position}] must be three strings")
                triples.append((triple[0], triple[1], triple[2]))

        hops = heuristic_hops(question, gold_titles)
        questions.append(PoolQuestion(
            id=question_id,
            question=question,
            type=question_type,
            level=level,
            answers=(answer,),
            paragraphs=tuple(
                Paragraph(title, text, title in hops, hops.get(title))
                for title, text in zip(titles, texts)
            ),
            supporting_facts=tuple(supporting),
            decomposition=(),
            triples=tuple(triples),
        ))
    return LoadResult(questions, dropped)


def load_hotpotqa(records: Any) -> LoadResult:
    return load_wiki_style(records, with_triples=False)


def load_2wiki(records: Any) -> LoadResult:
    return load_wiki_style(records, with_triples=True)


def load_musique(records: Any) -> LoadResult:
    if not isinstance(records, list):
        raise SourceFormatError("source: expected a list of questions")
    questions: list[PoolQuestion] = []
    dropped: Counter[str] = Counter()
    for index, raw in enumerate(records):
        where = f"record {index}"
        record = require_record(raw, where)
        question_id = require(record, "id", str, where)
        where = f"question {question_id}"
        if not require(record, "answerable", bool, where):
            dropped["unanswerable"] += 1
            continue
        question = require(record, "question", str, where)
        answer = require(record, "answer", str, where)
        aliases = require(record, "answer_aliases", list, where)
        hop_match = re.match(r"^(\d)hop", question_id)
        if not hop_match:
            raise SourceFormatError(f"{where}: id does not start with a hop count")

        paragraphs: dict[int, tuple[str, str, bool]] = {}
        for position, raw_paragraph in enumerate(require(record, "paragraphs", list, where)):
            paragraph_where = f"{where}: paragraphs[{position}]"
            paragraph = require_record(raw_paragraph, paragraph_where)
            paragraph_index = require(paragraph, "idx", int, paragraph_where)
            if paragraph_index in paragraphs:
                raise SourceFormatError(f"{paragraph_where}: duplicate idx {paragraph_index}")
            paragraphs[paragraph_index] = (
                require(paragraph, "title", str, paragraph_where),
                require(paragraph, "paragraph_text", str, paragraph_where).strip(),
                require(paragraph, "is_supporting", bool, paragraph_where),
            )

        hops: dict[int, int] = {}
        decomposition: list[dict[str, Any]] = []
        for step, raw_step in enumerate(require(record, "question_decomposition", list, where), start=1):
            step_where = f"{where}: question_decomposition[{step - 1}]"
            entry = require_record(raw_step, step_where)
            support = entry.get("paragraph_support_idx")
            if support is not None:
                if not isinstance(support, int) or support not in paragraphs:
                    raise SourceFormatError(f"{step_where}: paragraph_support_idx {support!r} is unknown")
                hops.setdefault(support, step)
            decomposition.append({
                "question": require(entry, "question", str, step_where),
                "answer": require(entry, "answer", str, step_where),
                "paragraphIndex": support,
            })
        if not any(gold for _, _, gold in paragraphs.values()):
            raise SourceFormatError(f"{where}: no supporting paragraphs")

        questions.append(PoolQuestion(
            id=question_id,
            question=question,
            type=f"{hop_match.group(1)}-hop",
            level=question_id.split("__", 1)[0],
            answers=(answer, *(alias for alias in aliases if isinstance(alias, str))),
            paragraphs=tuple(
                Paragraph(title, text, gold, hops.get(paragraph_index) if gold else None)
                for paragraph_index, (title, text, gold) in sorted(paragraphs.items())
            ),
            supporting_facts=(),
            decomposition=tuple(decomposition),
            triples=(),
        ))
    return LoadResult(questions, dropped)


LOADERS: dict[str, Callable[[Any], LoadResult]] = {
    "hotpotqa": load_hotpotqa,
    "musique": load_musique,
    "2wiki": load_2wiki,
}


def read_source(dataset: str, path: Path) -> Any:
    if dataset == "musique":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def drop_reason(question: PoolQuestion) -> str | None:
    gold = question.gold
    if len(gold) < 2:
        return "fewer-than-two-gold"
    if any(word_count(paragraph.title, paragraph.text) < MIN_WORDS for paragraph in gold):
        return "short-gold-paragraph"
    return None


def eligible_questions(questions: list[PoolQuestion], dropped: Counter[str]) -> list[PoolQuestion]:
    eligible = []
    for question in questions:
        reason = drop_reason(question)
        if reason:
            dropped[reason] += 1
        else:
            eligible.append(question)
    return eligible


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {name: total * count / population for name, count in counts.items()}
    result = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(result.values())
    for name in sorted(exact, key=lambda name: (result[name] - exact[name], name))[:remainder]:
        result[name] += 1
    return result


def sample_splits(questions: list[PoolQuestion], per_split: int, seed: int) -> dict[str, list[PoolQuestion]]:
    if per_split < 1:
        raise ValueError("--per-split must be at least 1")
    needed = per_split * len(SPLITS)
    if len(questions) < needed:
        raise ValueError(f"Only {len(questions)} eligible questions; {needed} are needed")
    generator = random.Random(seed)
    by_type: dict[str, list[PoolQuestion]] = defaultdict(list)
    for question in sorted(questions, key=lambda question: question.id):
        by_type[question.type].append(question)
    for name in sorted(by_type):
        generator.shuffle(by_type[name])
    split_quotas = quotas({name: len(group) for name, group in by_type.items()}, per_split)
    for name, quota in split_quotas.items():
        if quota * len(SPLITS) > len(by_type[name]):
            raise ValueError(f"Type {name!r} has too few questions for {len(SPLITS)} splits of {quota}")
    splits: dict[str, list[PoolQuestion]] = {}
    for position, split in enumerate(SPLITS):
        chosen = [
            question
            for name in sorted(by_type)
            for question in by_type[name][position * split_quotas[name]:(position + 1) * split_quotas[name]]
        ]
        generator.shuffle(chosen)
        splits[split] = chosen
    return splits


def paragraph_id(prefix: str, title: str, text: str) -> str:
    digest = hashlib.sha1("\0".join((title, text)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


@dataclass
class Pool:
    corpus: dict[str, dict[str, str]]
    qrels: dict[str, list[tuple[str, str]]]
    sidecar: dict[str, dict[str, Any]]
    triples: dict[str, list[list[str]]]
    short_distractors: int
    hop_known: int
    hop_unknown: int


def build_pool(splits: dict[str, list[PoolQuestion]], prefix: str) -> Pool:
    corpus: dict[str, dict[str, str]] = {}
    qrels: dict[str, list[tuple[str, str]]] = {split: [] for split in SPLITS}
    sidecar: dict[str, dict[str, Any]] = {}
    triples: dict[str, list[list[str]]] = {}
    short_distractors = hop_known = hop_unknown = 0
    for split in SPLITS:
        for question in splits[split]:
            gold_entries = []
            for paragraph in question.paragraphs:
                if not paragraph.gold and word_count(paragraph.title, paragraph.text) < MIN_WORDS:
                    short_distractors += 1
                    continue
                document_id = paragraph_id(prefix, paragraph.title, paragraph.text)
                corpus.setdefault(document_id, {"_id": document_id, "title": paragraph.title,
                                                "text": paragraph.text})
                if paragraph.gold:
                    qrels[split].append((question.id, document_id))
                    gold_entries.append({"id": document_id, "title": paragraph.title, "hop": paragraph.hop})
                    if paragraph.hop is None:
                        hop_unknown += 1
                    else:
                        hop_known += 1
            sidecar[question.id] = {
                "split": split,
                "type": question.type,
                "level": question.level,
                "answers": list(question.answers),
                "supportingFacts": [list(fact) for fact in question.supporting_facts],
                "decomposition": list(question.decomposition),
                "gold": gold_entries,
            }
            if question.triples:
                triples[question.id] = [list(triple) for triple in question.triples]
    overlap = set(sidecar) & set(corpus)
    if overlap:
        raise ValueError(f"Query IDs collide with corpus IDs: {sorted(overlap)[:5]}")
    return Pool(corpus, qrels, sidecar, triples, short_distractors, hop_known, hop_unknown)


def question_phrases(questions: list[PoolQuestion]) -> set[str]:
    phrases: set[str] = set()
    for question in questions:
        tokens = normalize(question.question).split()
        for start in range(len(tokens)):
            for end in range(start + 1, min(start + MAX_TITLE_WORDS, len(tokens)) + 1):
                phrases.add(" ".join(tokens[start:end]))
    return phrases


def add_filler(pool: Pool, sampled: list[PoolQuestion], filler: list[PoolQuestion],
               count: int, seed: int, prefix: str) -> int:
    pooled_titles = {variant for document in pool.corpus.values() for variant in title_variants(document["title"])}
    mentioned = question_phrases(sampled)
    candidates: dict[str, dict[str, str]] = {}
    for question in filler:
        for paragraph in question.paragraphs:
            variants = title_variants(paragraph.title)
            if variants & pooled_titles or variants & mentioned:
                continue
            if word_count(paragraph.title, paragraph.text) < MIN_WORDS:
                continue
            document_id = paragraph_id(prefix, paragraph.title, paragraph.text)
            if document_id not in pool.corpus:
                candidates.setdefault(document_id, {"_id": document_id, "title": paragraph.title,
                                                    "text": paragraph.text})
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} filler paragraphs are available; {count} requested")
    chosen = random.Random(seed).sample(sorted(candidates), count)
    for document_id in chosen:
        pool.corpus[document_id] = candidates[document_id]
    return len(chosen)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset(target: Path, pool: Pool, splits: dict[str, list[PoolQuestion]],
                  manifest: dict[str, Any], force: bool) -> None:
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / "qrels").mkdir()
        write_jsonl(staging / "corpus.jsonl", list(pool.corpus.values()))
        write_jsonl(staging / "queries.jsonl", [
            {"_id": question.id, "text": question.question}
            for split in SPLITS for question in splits[split]
        ])
        for split in SPLITS:
            with (staging / "qrels" / f"{split}.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["query-id", "corpus-id", "score"])
                writer.writerows([query_id, document_id, 1] for query_id, document_id in pool.qrels[split])
        write_json(staging / "multihop.json", {"builder": BUILDER_VERSION, "source": manifest["source"]["dataset"],
                                               "queries": pool.sidecar})
        if pool.triples:
            write_json(staging / "evidence-triples.json", pool.triples)
        write_json(staging / "manifest.json", manifest)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_name(name: str) -> str:
    if not DATASET_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid dataset name: {name}")
    if not name.startswith(NAME_PREFIX):
        raise ValueError(f"Multi-hop dataset names must start with '{NAME_PREFIX}': {name}")
    return name


def describe_source(dataset: str, path: Path) -> dict[str, str]:
    return {"dataset": dataset, "file": path.name, "sha256": file_sha256(path)}


def build(dataset: str, source: Path, name: str, per_split: int, seed: int, filler_count: int,
          filler_source: Path | None, force: bool, datasets_root: Path = DATASETS_ROOT) -> dict[str, Any]:
    validate_name(name)
    if (filler_count > 0) != (filler_source is not None):
        raise ValueError("--filler and --filler-source must be given together")
    loaded = LOADERS[dataset](read_source(dataset, source))
    dropped = Counter(loaded.dropped)
    eligible = eligible_questions(loaded.questions, dropped)
    splits = sample_splits(eligible, per_split, seed)
    prefix = ID_PREFIXES[dataset]
    pool = build_pool(splits, prefix)
    gold_count = len({document_id for rows in pool.qrels.values() for _, document_id in rows})
    filler_added = 0
    if filler_source is not None:
        filler = LOADERS[dataset](read_source(dataset, filler_source))
        sampled = [question for split in SPLITS for question in splits[split]]
        filler_added = add_filler(pool, sampled, filler.questions, filler_count, seed, prefix)
    titles: dict[str, set[str]] = defaultdict(set)
    for document in pool.corpus.values():
        titles[document["title"]].add(document["text"])
    manifest = {
        "builder": BUILDER_VERSION,
        "name": name,
        "kind": "multi-hop",
        "source": describe_source(dataset, source),
        "fillerSource": describe_source(dataset, filler_source) if filler_source else None,
        "seed": seed,
        "perSplit": per_split,
        "loadedQuestions": len(loaded.questions) + sum(loaded.dropped.values()),
        "droppedQuestions": dict(sorted(dropped.items())),
        "eligibleQuestions": len(eligible),
        "typeCounts": {
            "eligible": dict(sorted(Counter(question.type for question in eligible).items())),
            **{split: dict(sorted(Counter(question.type for question in splits[split]).items()))
               for split in SPLITS},
        },
        "corpus": {
            "paragraphs": len(pool.corpus),
            "goldParagraphs": gold_count,
            "fillerParagraphs": filler_added,
            "shortDistractorsSkipped": pool.short_distractors,
            "titlesWithSeveralTexts": sum(len(texts) > 1 for texts in titles.values()),
        },
        "hopRoles": {"known": pool.hop_known, "unknown": pool.hop_unknown},
        "splits": {split: [question.id for question in splits[split]] for split in SPLITS},
    }
    write_dataset(datasets_root / name, pool, splits, manifest, force)
    return manifest


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pooled multi-hop corpus in BEIR layout.")
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--per-split", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filler", type=int, default=0)
    parser.add_argument("--filler-source", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments()
    manifest = build(arguments.dataset, arguments.source, arguments.name, arguments.per_split,
                     arguments.seed, arguments.filler, arguments.filler_source, arguments.force)
    corpus = manifest["corpus"]
    print(f"{manifest['name']}: {corpus['paragraphs']} paragraphs "
          f"({corpus['goldParagraphs']} gold, {corpus['fillerParagraphs']} filler); "
          f"{manifest['perSplit']} questions per split; dropped {manifest['droppedQuestions']}; "
          f"hop roles {manifest['hopRoles']}")


if __name__ == "__main__":
    main()
