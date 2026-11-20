import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const benchRoot = dirname(fileURLToPath(import.meta.url));
const venv = join(benchRoot, '..', 'beir', '.venv');
const python =
	process.platform === 'win32' ? join(venv, 'Scripts', 'python.exe') : join(venv, 'bin', 'python');

if (!existsSync(python)) {
	console.error(
		`The BEIR harness venv is missing (${python}). Create it as described in benchmarks/beir/README.md.`
	);
	process.exit(1);
}

const bench = spawn(python, [join(benchRoot, 'bench.py'), ...process.argv.slice(2)], {
	stdio: 'inherit'
});

// Ctrl+C and Ctrl+Break reach the whole console, bench included; staying alive until it exits
// lets it stop a running job first. A stop aimed at this launcher alone is passed on.
for (const signal of ['SIGINT', 'SIGBREAK']) process.on(signal, () => {});
for (const signal of ['SIGTERM', 'SIGHUP']) process.on(signal, () => bench.kill(signal));

bench.on('error', (error) => {
	console.error(`Could not start ${python}: ${error.message}`);
	process.exit(1);
});
bench.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
