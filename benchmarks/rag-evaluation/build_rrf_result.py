"""Build an application RRF ranking from an existing complete saved run."""
from __future__ import annotations
import argparse, copy, json, random, sys
from pathlib import Path
from rrf import reciprocal_rank_fusion

BEIR_ROOT = Path(__file__).resolve().parents[1] / 'beir'
sys.path.insert(0, str(BEIR_ROOT))
from run import evaluate_method  # noqa: E402
from beir.datasets.data_loader import GenericDataLoader  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite existing artifact: {args.output}')
    source = json.loads(args.rankings.read_text())
    mapping = json.loads(args.mapping.read_text())['applicationToBeir'] if args.mapping else {}
    if not all(key in source for key in ('bm25', 'semantic')):
        raise SystemExit('saved rankings must contain bm25 and semantic')
    query_ids = sorted(set(source['bm25']) & set(source['semantic']), key=lambda x: int(x) if x.isdigit() else x)
    if len(query_ids) != len(source['bm25']) or len(query_ids) != len(source['semantic']):
        raise SystemExit('bm25 and semantic query coverage differs')
    rankings = {}
    for query_id in query_ids:
        lists = []
        for method in ('bm25', 'semantic'):
            entries = source[method][query_id]
            # Saved application rankings use larger artificial scores for better ranks.
            lists.append([mapping.get(doc, doc) for doc, _rank in sorted(entries.items(), key=lambda item: item[1], reverse=True)])
        rankings[query_id] = [{'docId': doc, 'score': score} for doc, score in reciprocal_rank_fusion(lists, k=60, limit=args.limit)]
    dataset_root = BEIR_ROOT / 'datasets' / 'scifact'
    _corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_root)).load(split='test')
    expected_qids = set(queries) & set(qrels)
    if set(query_ids) != expected_qids:
        raise SystemExit(f'query ID mismatch: rankings={len(query_ids)} qrels={len(expected_qids)}')
    results = {qid: {item['docId']: item['score'] for item in items} for qid, items in rankings.items()}
    metrics = evaluate_method(qrels, results)
    per_query = {}
    for qid in query_ids:
        one = evaluate_method({qid: qrels[qid]}, {qid: results[qid]})
        per_query[qid] = {metric: values.get('NDCG@10', values.get('MRR@10', values.get('MAP@10', values.get('Recall@10', values.get('P@10'))))) for metric, values in one.items()}
    rng = random.Random(42)
    ids = query_ids
    def mean(sample, metric): return sum(per_query[q][metric] for q in sample) / len(sample)
    ci = {}
    for metric in ('ndcg', 'mrr', 'map', 'recall', 'precision'):
        samples = sorted(mean([rng.choice(ids) for _ in ids], metric) for _ in range(1000))
        ci[metric] = {'mean': mean(ids, metric), 'lower95': samples[25], 'upper95': samples[974], 'resamples': 1000, 'seed': 42}
    result = {'method': 'application-bm25-semantic-rrf', 'rrfConstant': 60, 'limit': args.limit,
              'dataset': 'scifact', 'split': 'test', 'queryCount': len(query_ids), 'qrelsCount': len(qrels),
              'source': str(args.rankings), 'metrics': metrics, 'bootstrap95': ci, 'perQuery': per_query, 'rankings': rankings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(args.output)
    print(json.dumps({'output': str(args.output), 'queryCount': len(query_ids)}))
    return 0
if __name__ == '__main__': raise SystemExit(main())
