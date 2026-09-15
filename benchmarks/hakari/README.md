# HAKARI integration scaffold

This adapter is intentionally disabled until the optional HAKARI dependency is
installed and a loopback-only TypeScript sidecar has passed parity tests. The
sidecar must accept one query and a fixed list of `{id,text}` candidates and
return raw scores plus ranked IDs. It must call the existing TypeScript
`rerankCandidates` implementation; normal `/search` is not a valid reranker
comparison because it creates its own candidates.

Install later in a separate Python 3.12/uv environment:

```bash
uv add "hakari-bench @ git+https://github.com/hakari-bench/hakari-bench.git"
```

No HAKARI score is reported by this scaffold.

## Safe execution

Run HAKARI through `python benchmarks/hakari/run_locked.py <script>` so it shares
the global benchmark lease with BEIR. The wrapper starts the script in an owned
process group, includes all descendants in its memory total, and cleans up that
group on memory pressure, timeout, or interruption. Its defaults are a 6 GiB RSS
warning, a 7 GiB RSS stop, and a one-hour timeout. Status and memory samples are
written under `.cache/hakari/`.

The raw `sidecar.ts` remains disabled by default and must not be launched manually
during another benchmark. Inspect ownership with
`python benchmarks/rag-evaluation/full_beir_plan.py --status`; stale cleanup only
terminates a fully verified owner (PID, start identity, command, and process group).
