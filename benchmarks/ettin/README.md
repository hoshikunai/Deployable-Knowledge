# Ettin loopback reranker

This opt-in service hosts an Ettin reranker through
SentenceTransformers' PyTorch backend on CPU. It is intentionally loopback-only and is
not started by the application. Install `requirements.txt`, then run:

Python 3.10 or newer is required. SentenceTransformers 5.4.1 or newer is
required because Ettin uses the modular CrossEncoder architecture introduced
in that release. Ettin's tokenizer also requires Transformers 5; the model
was published with version 5.7.0, pinned in `requirements.txt`. Re-run the
requirements installation in an existing Ettin virtual environment before
starting the service. The ONNX extra is not used because its current
Optimum-ONNX dependency requires Transformers 4, which cannot load Ettin's
Transformers 5 tokenizer.

```bash
.cache/beir/ettin-venv/bin/python -m pip install -r benchmarks/ettin/requirements.txt
.cache/beir/ettin-venv/bin/python benchmarks/ettin/service.py
```

Use `HF_HUB_OFFLINE=1` only after the Ettin model and tokenizer have been
downloaded into the Hugging Face cache; omit it on the first successful load.

The service supports `ettin-32m`, `ettin-68m`, `ettin-150m`, and `ettin-400m`; the
default is `ettin-32m`. Select the same model in both processes, for example:

```bash
RAG_RERANK_MODEL=ettin-68m .cache/beir/ettin-venv/bin/python benchmarks/ettin/service.py
```

```bash
export RAG_RERANK_MODEL=ettin-68m
export RAG_ETTIN_ENDPOINT=http://127.0.0.1:41792/rerank
```

On the first launch of a model, omit `HF_HUB_OFFLINE=1` so Hugging Face can download
the model and tokenizer. After the files are cached, you may use:

```bash
HF_HUB_OFFLINE=1 RAG_RERANK_MODEL=ettin-68m \
  .cache/beir/ettin-venv/bin/python benchmarks/ettin/service.py
```

The CPU scorer defaults to two inference threads; set `ETTIN_CPU_THREADS` in
the service process to change that limit after measuring memory and speed.
The application fails explicitly if the service is unavailable or returns an
invalid score vector; it never silently falls back to MS MARCO.

The service returns its exact model ID and the application rejects a mismatch, but
benchmark provenance is not yet persisted automatically; record the selected model and
backend alongside any benchmark artifact before comparing runs.
