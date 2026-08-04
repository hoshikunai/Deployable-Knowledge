import assert from 'node:assert/strict';
import test from 'node:test';
import { Ollama } from './ollama';
import type { ProviderChatOptions } from './provider';

test('Ollama maps reasoning settings into its chat request', async (t) => {
	await t.test('zero disables thinking without changing the answer budget', async () => {
		const request = await captureChatRequest({ maxTokens: 1024, reasoningBudget: 0 });

		assert.equal(request.think, false);
		assert.equal(request.options.num_predict, 1024);
	});

	await t.test('a positive budget enables thinking and reserves answer headroom', async () => {
		const request = await captureChatRequest({ maxTokens: 1024, reasoningBudget: 512 });

		assert.equal(request.think, true);
		assert.equal(request.options.num_predict, 1536);
	});

	await t.test('an omitted budget preserves the model default', async () => {
		const request = await captureChatRequest({ maxTokens: 1024 });

		assert.ok(!Object.hasOwn(request, 'think'));
		assert.equal(request.options.num_predict, 1024);
	});
});

type OllamaRequestBody = {
	think?: boolean;
	options: {
		num_predict?: number;
	};
};

async function captureChatRequest(options: ProviderChatOptions): Promise<OllamaRequestBody> {
	const originalFetch = globalThis.fetch;
	let requestBody: OllamaRequestBody | undefined;

	globalThis.fetch = async (input) => {
		const request = input instanceof Request ? input : new Request(input);
		requestBody = (await request.json()) as OllamaRequestBody;

		return new Response(
			`${JSON.stringify({ message: { role: 'assistant', content: 'Answer' }, done: false })}\n${JSON.stringify({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' })}\n`,
			{ status: 200, headers: { 'Content-Type': 'application/x-ndjson' } }
		);
	};

	try {
		const provider = new Ollama();
		for await (const _chunk of provider.streamChat(
			[{ role: 'user', content: 'Question' }],
			'test-model',
			options
		)) {
			// Consume the complete response so request and stream behavior are exercised.
		}
	} finally {
		globalThis.fetch = originalFetch;
	}

	assert.ok(requestBody);
	return requestBody;
}
