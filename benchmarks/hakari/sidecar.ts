/** Benchmark-only loopback reranker sidecar. Disabled unless explicitly enabled. */
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
	rerankCandidates,
	type RerankCandidate
} from '../../src/lib/server/rag/search/cross-rerank';

type Payload = { query: string; candidates: RerankCandidate[] };
const enabled = process.env.BENCHMARK_RERANKER_ENABLED === '1';
const host = '127.0.0.1';
const port = Number(process.env.BENCHMARK_RERANKER_PORT ?? 41791);

function reply(response: ServerResponse, status: number, body: unknown) {
	response.writeHead(status, { 'content-type': 'application/json' });
	response.end(JSON.stringify(body));
}

async function readBody(request: IncomingMessage): Promise<unknown> {
	let raw = '';
	for await (const chunk of request) raw += chunk;
	if (raw.length > 2_000_000) throw new Error('request too large');
	return JSON.parse(raw);
}

function validate(value: unknown): Payload {
	if (!value || typeof value !== 'object') throw new Error('payload must be an object');
	const payload = value as Partial<Payload>;
	if (typeof payload.query !== 'string' || !payload.query.trim())
		throw new Error('query must be non-empty');
	if (!Array.isArray(payload.candidates) || payload.candidates.length > 256)
		throw new Error('candidates must contain at most 256 items');
	const candidates = payload.candidates.map((candidate) => {
		if (!candidate || typeof candidate !== 'object') throw new Error('invalid candidate');
		const item = candidate as Partial<RerankCandidate>;
		if (typeof item.chunkId !== 'string' || !item.chunkId || typeof item.content !== 'string')
			throw new Error('candidate requires chunkId and content');
		return { chunkId: item.chunkId, content: item.content };
	});
	return { query: payload.query, candidates };
}

if (!enabled) {
	console.error('Benchmark reranker disabled. Set BENCHMARK_RERANKER_ENABLED=1 explicitly.');
	process.exitCode = 2;
} else {
	const server = createServer(async (request, response) => {
		if (request.method !== 'POST' || request.url !== '/rerank')
			return reply(response, 404, { error: 'not found' });
		try {
			const payload = validate(await readBody(request));
			const ranked = await rerankCandidates(payload.query, payload.candidates);
			reply(response, 200, {
				ranked: ranked.map((item, rank) => ({
					id: item.chunkId,
					score: item.relevance,
					rank: rank + 1
				}))
			});
		} catch (error) {
			reply(response, 400, { error: error instanceof Error ? error.message : 'invalid request' });
		}
	});
	server.listen(port, host, () =>
		console.log(JSON.stringify({ host, port, pid: process.pid, benchmarkOwned: true }))
	);
	const stop = () => server.close(() => process.exit(0));
	process.once('SIGINT', stop);
	process.once('SIGTERM', stop);
}
