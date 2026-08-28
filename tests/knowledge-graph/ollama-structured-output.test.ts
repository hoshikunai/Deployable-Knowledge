import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { Ollama } from '../../src/lib/server/providers/ollama';

describe('Ollama structured output transport', () => {
	it('disables streaming and exposes completion diagnostics', async () => {
		const originalFetch = globalThis.fetch;
		const requestBodies: Record<string, unknown>[] = [];
		globalThis.fetch = async (input) => {
			const request = input instanceof Request ? input : new Request(input);
			requestBodies.push(JSON.parse(await request.text()) as Record<string, unknown>);
			return new Response(
				JSON.stringify({
					message: { role: 'assistant', content: '{"ok":true}', thinking: '' },
					done: true,
					done_reason: 'stop',
					prompt_eval_count: 123,
					eval_count: 17
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } }
			);
		};

		try {
			const chunks = [];
			for await (const chunk of new Ollama().streamChat(
				[{ role: 'user', content: 'Return JSON.' }],
				'test-model',
				{
					thinking: false,
					structuredOutput: {
						type: 'object',
						properties: { ok: { type: 'boolean' } },
						required: ['ok']
					}
				}
			)) {
				chunks.push(chunk);
			}

			assert.equal(requestBodies[0]?.stream, false);
			assert.equal(requestBodies[0]?.think, false);
			assert.deepEqual(chunks, [
				{
					content: '{"ok":true}',
					finishReason: 'stop',
					inputTokens: 123,
					outputTokens: 17
				}
			]);
		} finally {
			globalThis.fetch = originalFetch;
		}
	});
});
