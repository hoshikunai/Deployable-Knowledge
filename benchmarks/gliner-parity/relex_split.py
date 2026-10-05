from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import torch
from torch import nn
from gliner import GLiNER
from gliner.modeling.outputs import GLiNERRelexOutput
from gliner.modeling.utils import extract_prompt_features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "gliner-parity"
SPLIT_DIR = OUT / "relex-split"
RELEX_REPO = "knowledgator/gliner-relex-large-v0.5"
THREADS = max(1, min(8, (os.cpu_count() or 4) // 4))
ENCODER_INPUTS = ("input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask")
OPSET = 19


def exportable_diag_embed(values: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(values.size(-1), dtype=values.dtype, device=values.device)
    return values.unsqueeze(-1) * identity


class EncoderGraph(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask):
        core = self.core
        token_embeds = core.token_rep_layer(input_ids, attention_mask)
        prompts, _, words, mask = core._extract_prompt_features_and_word_embeddings(
            token_embeds, input_ids, attention_mask, text_lengths, words_mask)
        words = core.rnn(words, mask)
        words, mask = core._fit_length(words, mask, span_idx.size(1) // core.config.max_width)
        prompts = core.prompt_rep_layer(prompts)
        span_rep = core.span_rep_layer(words, span_idx * span_mask.unsqueeze(-1).long())
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts)
        rel_prompts, _ = extract_prompt_features(core.config.rel_token_index, token_embeds, input_ids,
                                                 attention_mask, prompts.size(0), prompts.size(2),
                                                 core.config.embed_rel_token)
        return scores, span_rep, rel_prompts


class AdjacencyGraph(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.layer = core.relations_rep_layer

    def forward(self, entity_rep, entity_mask):
        return self.layer(entity_rep, entity_mask)


class PairGraph(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.layer = core.pair_rep_layer

    def forward(self, head_rep, tail_rep, rel_prompts):
        pair_rep = self.layer(torch.cat((head_rep, tail_rep), dim=-1))
        return torch.einsum("BND,BCD->BNC", pair_rep, rel_prompts)


def load_model() -> GLiNER:
    torch.set_num_threads(THREADS)
    model = GLiNER.from_pretrained(RELEX_REPO).to("cpu")
    model.eval()
    return model


def sample_batch(model: GLiNER) -> dict[str, torch.Tensor]:
    texts = json.loads((OUT / "texts.json").read_text(encoding="utf-8"))
    prepared = model.prepare_batch([texts["chunks"][0]["text"]], texts["entityLabels"],
                                   relations=texts["relations"])
    return model.collate_batch(prepared["input_x"], prepared["entity_types"],
                               relation_types=prepared["relation_types"], label_names=prepared["label_names"])


def export(target: Path) -> None:
    torch.diag_embed = exportable_diag_embed
    target.mkdir(parents=True, exist_ok=True)
    model = load_model()
    model.data_processor.transformer_tokenizer.save_pretrained(target)
    model.config.to_json_file(target / "gliner_config.json")
    core = model.model
    batch = sample_batch(model)
    encoder_args = tuple(batch[name] for name in ENCODER_INPUTS)
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    scores, span_rep, rel_prompts = EncoderGraph(core)(*encoder_args)
    torch.onnx.export(
        EncoderGraph(core).eval(), encoder_args, str(target / "encoder.onnx"),
        input_names=list(ENCODER_INPUTS), output_names=["scores", "span_rep", "rel_prompts"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
            "words_mask": {0: "batch", 1: "tokens"}, "text_lengths": {0: "batch"},
            "span_idx": {0: "batch", 1: "spans"}, "span_mask": {0: "batch", 1: "spans"},
            "scores": {0: "batch", 1: "words", 3: "entity_classes"}, "span_rep": {0: "batch", 1: "words"},
            "rel_prompts": {0: "batch", 1: "relation_classes"},
        },
        opset_version=OPSET, dynamo=False,
    )
    dimension = span_rep.size(-1)
    entities = torch.randn(1, 5, dimension)
    torch.onnx.export(
        AdjacencyGraph(core).eval(), (entities, torch.ones(1, 5, dtype=torch.long)),
        str(target / "adjacency.onnx"), input_names=["entity_rep", "entity_mask"], output_names=["adjacency"],
        dynamic_axes={"entity_rep": {0: "batch", 1: "entities"}, "entity_mask": {0: "batch", 1: "entities"},
                      "adjacency": {0: "batch", 1: "entities", 2: "entities_2"}},
        opset_version=OPSET, dynamo=False,
    )
    torch.onnx.export(
        PairGraph(core).eval(), (torch.randn(1, 7, dimension), torch.randn(1, 7, dimension), rel_prompts),
        str(target / "pairs.onnx"), input_names=["head_rep", "tail_rep", "rel_prompts"],
        output_names=["rel_logits"],
        dynamic_axes={"head_rep": {0: "batch", 1: "pairs"}, "tail_rep": {0: "batch", 1: "pairs"},
                      "rel_prompts": {0: "batch", 1: "relation_classes"},
                      "rel_logits": {0: "batch", 1: "pairs", 2: "relation_classes"}},
        opset_version=OPSET, dynamo=False,
    )
    sizes = {path.name: f"{path.stat().st_size / 1e6:.0f} MB" for path in sorted(target.glob("*.onnx"))}
    print(f"exported in {time.perf_counter() - started:.0f}s: {sizes}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def select_entities(scores: np.ndarray, span_rep: np.ndarray, span_idx: np.ndarray, span_mask: np.ndarray,
                    threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, dimension = scores.shape[0], span_rep.shape[-1]
    if scores.shape[-1] > 0:
        keep = (sigmoid(scores).max(axis=-1).reshape(batch, -1) > threshold) & span_mask.reshape(batch, -1).astype(bool)
    else:
        keep = np.zeros(span_mask.reshape(batch, -1).shape, dtype=bool)
    representations = span_rep.reshape(batch, -1, dimension)
    boundaries = (span_idx * span_mask[..., None]).reshape(batch, -1, 2)
    count = int(keep.sum(axis=1).max(initial=0))
    entity_rep = np.zeros((batch, count, dimension), dtype=np.float32)
    entity_mask = np.zeros((batch, count), dtype=np.int64)
    entity_spans = np.zeros((batch, count, 2), dtype=np.int64)
    for row in range(batch):
        positions = np.flatnonzero(keep[row])
        entity_rep[row, :len(positions)] = representations[row, positions]
        entity_mask[row, :len(positions)] = 1
        entity_spans[row, :len(positions)] = boundaries[row, positions]
    return entity_rep, entity_mask, entity_spans


def build_pairs(adjacency: np.ndarray, entity_rep: np.ndarray,
                threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch, count = adjacency.shape[0], adjacency.shape[1]
    dimension = entity_rep.shape[-1]
    rows, cols = np.meshgrid(np.arange(count), np.arange(count), indexing="ij")
    off_diagonal = rows != cols
    rows, cols = rows[off_diagonal], cols[off_diagonal]
    selected = [np.stack([rows[adjacency[row, rows, cols] > threshold],
                          cols[adjacency[row, rows, cols] > threshold]], axis=-1) for row in range(batch)]
    pairs = max((item.shape[0] for item in selected), default=0)
    if pairs == 0:
        zeros = np.zeros((batch, 1, dimension), dtype=np.float32)
        return np.full((batch, 1, 2), -1, dtype=np.int64), np.zeros((batch, 1), dtype=bool), zeros, zeros
    pair_idx = np.full((batch, pairs, 2), -1, dtype=np.int64)
    pair_mask = np.zeros((batch, pairs), dtype=bool)
    for row, item in enumerate(selected):
        pair_idx[row, :len(item)] = item
        pair_mask[row, :len(item)] = True
    gather = np.clip(pair_idx, 0, None)
    batch_index = np.arange(batch)[:, None]
    return pair_idx, pair_mask, entity_rep[batch_index, gather[..., 0]], entity_rep[batch_index, gather[..., 1]]


class SplitRunner:
    def __init__(self, sessions: dict[str, Any] | None = None, core: nn.Module | None = None) -> None:
        self.sessions = sessions
        self.core = core

    def run(self, name: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.sessions is not None:
            return self.sessions[name].run(None, feeds)
        modules = {"encoder": EncoderGraph, "adjacency": AdjacencyGraph, "pairs": PairGraph}
        with torch.inference_mode():
            outputs = modules[name](self.core)(*(torch.from_numpy(value) for value in feeds.values()))
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return [output.numpy() for output in outputs]

    def forward(self, batch: dict[str, Any], threshold: float, adjacency_threshold: float) -> dict[str, np.ndarray]:
        feeds = {name: np.ascontiguousarray(batch[name].numpy()) for name in ENCODER_INPUTS}
        scores, span_rep, rel_prompts = self.run("encoder", feeds)
        entity_rep, entity_mask, entity_spans = select_entities(scores, span_rep, feeds["span_idx"],
                                                                feeds["span_mask"], threshold)
        if entity_rep.shape[1] > 0:
            adjacency = self.run("adjacency", {"entity_rep": entity_rep, "entity_mask": entity_mask})[0]
        else:
            adjacency = np.zeros((entity_rep.shape[0], 0, 0), dtype=np.float32)
        pair_idx, pair_mask, head_rep, tail_rep = build_pairs(adjacency, entity_rep, adjacency_threshold)
        rel_logits = self.run("pairs", {"head_rep": np.ascontiguousarray(head_rep, dtype=np.float32),
                                        "tail_rep": np.ascontiguousarray(tail_rep, dtype=np.float32),
                                        "rel_prompts": rel_prompts})[0]
        return {"logits": scores, "rel_idx": pair_idx, "rel_logits": rel_logits, "rel_mask": pair_mask,
                "entity_spans": entity_spans}


def onnx_sessions() -> dict[str, onnxruntime.InferenceSession]:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return {name: onnxruntime.InferenceSession(str(SPLIT_DIR / f"{name}.onnx"), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            for name in ("encoder", "adjacency", "pairs")}


def route_through(model: GLiNER, runner: SplitRunner) -> None:
    def run_batch(batch, threshold=0.5, adjacency_threshold=0.5, **_):
        result = runner.forward(batch, threshold, adjacency_threshold)
        return GLiNERRelexOutput(**{name: torch.from_numpy(np.ascontiguousarray(value))
                                    for name, value in result.items()})

    model.run_batch = run_batch


def tensors() -> int:
    model = load_model()
    core = model.model
    texts = json.loads((OUT / "texts.json").read_text(encoding="utf-8"))
    runners = {"split-torch": SplitRunner(core=core), "split-onnx": SplitRunner(sessions=onnx_sessions())}
    worst: dict[str, dict[str, float]] = {name: {"logits": 0.0, "rel_logits": 0.0, "mismatches": 0}
                                           for name in runners}
    cases = [chunk["text"] for chunk in texts["chunks"]] + [
        "It rained for most of the afternoon and the streets stayed quiet."]
    for text in cases:
        prepared = model.prepare_batch([text], texts["entityLabels"], relations=texts["relations"])
        batch = model.collate_batch(prepared["input_x"], prepared["entity_types"],
                                    relation_types=prepared["relation_types"], label_names=prepared["label_names"])
        with torch.inference_mode():
            reference = core(**{name: batch[name] for name in ENCODER_INPUTS}, threshold=0.5, adjacency_threshold=0.5)
        expected = {name: getattr(reference, name).numpy() for name in
                    ("logits", "rel_idx", "rel_logits", "rel_mask", "entity_spans")}
        for name, runner in runners.items():
            got = runner.forward(batch, 0.5, 0.5)
            same_structure = all(np.array_equal(expected[key], got[key])
                                 for key in ("rel_idx", "rel_mask", "entity_spans"))
            worst[name]["mismatches"] += 0 if same_structure else 1
            worst[name]["logits"] = max(worst[name]["logits"], float(np.abs(expected["logits"] - got["logits"]).max()))
            if same_structure:
                valid = expected["rel_mask"]
                if valid.any():
                    delta = np.abs(expected["rel_logits"][valid] - got["rel_logits"][valid]).max()
                    worst[name]["rel_logits"] = max(worst[name]["rel_logits"], float(delta))
    print("| Runner | Cases with different entities or pairs | Max entity logit diff | Max relation logit diff |")
    print("|---|---|---|---|")
    for name, row in worst.items():
        print(f"| {name} | {row['mismatches']} of {len(cases)} | {row['logits']:.2e} | {row['rel_logits']:.2e} |")
    return 0 if all(row["mismatches"] == 0 for row in worst.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GLiNER-relex as three graphs and check them.")
    parser.add_argument("action", choices=("export", "tensors"))
    parser.add_argument("--out", type=Path, default=SPLIT_DIR)
    arguments = parser.parse_args()
    if arguments.action == "export":
        export(arguments.out.resolve())
        return 0
    return tensors()


if __name__ == "__main__":
    raise SystemExit(main())
