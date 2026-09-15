# RAG evaluation orchestration

This directory contains the safe, resumable coordination layer for the BEIR and
HAKARI work. It does not replace `benchmarks/beir`; existing run artifacts remain
immutable. The default command is a dry plan:

```bash
python benchmarks/rag-evaluation/orchestrator.py plan
python benchmarks/rag-evaluation/orchestrator.py inventory
python benchmarks/rag-evaluation/orchestrator.py report
```

`run` is intentionally conservative: it only prepares provenance and plans
work. Dataset execution remains an explicit, separately gated operation because
it can take hours and requires a verified isolated runtime.

The HAKARI adapter is in `benchmarks/hakari/`. It is disabled until HAKARI and
the loopback TypeScript sidecar have been independently validated.
