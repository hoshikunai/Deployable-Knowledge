# Benchmark memory pause root cause

The WSL dynamic memory maximum was 13312MB. Journald recorded pressure at 13:50:56;
the global OOM occurred at 13:58:36 after all 8GiB of swap was exhausted. PID 17670
(`MainThread`) had anon RSS 7,613,980kB; concurrent `MainThread` processes included
approximately 2.8GB and 735MB RSS. The surviving evidence does not establish exact
executable identity, but the grouping is consistent with overlapping interrupted
benchmark/model stacks. The user did not request concurrency; this was an
orchestration cleanup/ownership failure.

The prior runners had no shared OS lease. `full_beir_plan.py` used an alarm around
`run_suite.run_dataset`, while `run_suite.py` launched a server and a separate
`start_new_session` harness without recording ownership or supervising the server's
descendants. Interrupted agent/tool execution could therefore leave both trees alive.
Cleanup only waited on the direct `Popen` objects and port probes treated an occupied
port as a generic error; neither verified PID start identity, process groups, or total
RSS. The WSL OOM (8 GiB swap exhausted; one process ~7.6 GiB RSS and other model
processes ~2.8 GiB/~735 MiB) is consistent with those orphaned trees accumulating.

Safeguards now use `.cache/benchmark-lease.lock` (fcntl held for the run), metadata in
`.cache/benchmark-lease.json`, process-tree RSS aggregation, conservative preflight
headroom (4 GiB RAM/2 GiB swap), and TERM/KILL process-group cleanup.
BEIR and HAKARI use the same lease, so they cannot run concurrently. Both runners
monitor their complete owned process trees continuously and record an explicit
`memory-paused`, `timed-out`, or `interrupted` status before cleanup.
