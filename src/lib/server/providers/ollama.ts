import {
	Provider,
	type ProviderChatChunk,
	type ProviderChatMessage,
	type ProviderChatOptions
} from './provider';
import { createChatCodec } from './chat-codec';
import { readObject } from '$lib/server/utils/values';

const LLAMA_API_URL = 'http://localhost:11434';
const chatCodec = createChatCodec({
	assistantNullContent: 'empty',
	reasoningField: 'thinking',
	toolArguments: 'json',
	toolCallChunks: 'snapshot',
	toolResultNameField: 'tool_name'
});

export class Ollama extends Provider {
	override id = 'ollama';
	override name = 'Ollama';
	override apiKeyRequired = false;

	override async *streamChat(
		messages: ProviderChatMessage[],
		model: string,
		options: ProviderChatOptions = {}
	): AsyncGenerator<ProviderChatChunk> {
		const tools = options.toolChoice === 'none' ? undefined : options.tools;
		const think = resolveOllamaThinking(options.reasoningBudget);
		const numPredict = resolveOllamaMaxTokens(options.maxTokens, options.reasoningBudget);
		const req = new Request(`${LLAMA_API_URL}/api/chat`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model,
				messages: messages.map(chatCodec.encodeMessage),
				...(tools?.length ? { tools } : {}),
				...(think === undefined ? {} : { think }),
				options: {
					temperature: options.temperature,
					top_k: options.topK,
					num_predict: numPredict
				},
				stream: true
			}),
			signal: options.signal
		});

		const resp = await fetch(req);

		if (!resp.ok) {
			throw new Error(`Ollama chat failed (${resp.status}): ${await resp.text()}`);
		}

		const reader = resp.body?.getReader();
		const decoder = new TextDecoder();

		if (!reader) throw new Error('reader could not be created.');

		let buffer = '';

		while (true) {
			const { done, value } = await reader.read();
			buffer += decoder.decode(value, { stream: !done });
			const lines = buffer.split('\n');
			buffer = lines.pop() ?? '';

			for (const line of lines) {
				if (!line.trim()) continue;
				const data = readObject(JSON.parse(line) as unknown);
				const chunk = chatCodec.decodeChunk(data.message);

				if (chunk) yield chunk;
			}

			if (done) break;
		}

		if (buffer.trim()) {
			const data = readObject(JSON.parse(buffer) as unknown);
			const chunk = chatCodec.decodeChunk(data.message);

			if (chunk) yield chunk;
		}

		reader.releaseLock();
	}

	override async listModels(): Promise<string[]> {
		const req = new Request(`${LLAMA_API_URL}/api/tags`, {
			method: 'GET'
		});

		const resp = await fetch(req);
		const data = readObject(await resp.json());
		const models = Array.isArray(data.models) ? data.models : [];

		return models.flatMap((value) => {
			const model = readObject(value).model;
			return typeof model === 'string' ? [model] : [];
		});
	}
}

function resolveOllamaThinking(reasoningBudget: number | undefined): boolean | undefined {
	if (reasoningBudget === undefined) return undefined;
	return reasoningBudget !== 0;
}

function resolveOllamaMaxTokens(
	maxTokens: number | undefined,
	reasoningBudget: number | undefined
): number | undefined {
	if (maxTokens === undefined || reasoningBudget === undefined || reasoningBudget <= 0) {
		return maxTokens;
	}

	// Ollama's boolean `think` control cannot enforce a numeric thought budget.
	// Reserve that allowance as additional generation headroom so thinking does
	// not consume the entire visible-answer budget, matching the local provider.
	return Math.min(Number.MAX_SAFE_INTEGER, maxTokens + reasoningBudget);
}
