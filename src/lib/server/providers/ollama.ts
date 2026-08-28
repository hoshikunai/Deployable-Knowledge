import {
	Provider,
	type ProviderChatChunk,
	type ProviderChatMessage,
	type ProviderChatOptions
} from './provider';
import { cachedCapability } from './capability-cache';
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
		const stream = !options.structuredOutput;
		const req = new Request(`${LLAMA_API_URL}/api/chat`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model,
				messages: messages.map(chatCodec.encodeMessage),
				...(tools?.length ? { tools } : {}),
				...(options.structuredOutput ? { format: options.structuredOutput } : {}),
				...(options.thinking === undefined ? {} : { think: options.thinking }),
				options: {
					temperature: options.temperature,
					top_k: options.topK,
					num_predict: options.maxTokens,
					...(options.contextSize ? { num_ctx: options.contextSize } : {})
				},
				stream
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
				const chunk = decodeResponseChunk(data);

				if (chunk) yield chunk;
			}

			if (done) break;
		}

		if (buffer.trim()) {
			const data = readObject(JSON.parse(buffer) as unknown);
			const chunk = decodeResponseChunk(data);

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

	override supportsTools(model: string): Promise<boolean> {
		return cachedCapability(`ollama:${model}`, async () => {
			const resp = await fetch(`${LLAMA_API_URL}/api/show`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model }),
				signal: AbortSignal.timeout(2500)
			});
			if (!resp.ok) return true;
			const data = readObject(await resp.json());
			if (!Array.isArray(data.capabilities)) return true;
			return data.capabilities.includes('tools');
		});
	}
}

function decodeResponseChunk(data: Record<string, unknown>): ProviderChatChunk | null {
	const message = chatCodec.decodeChunk(data.message);
	const finishReason = typeof data.done_reason === 'string' ? data.done_reason : undefined;
	const inputTokens = integer(data.prompt_eval_count);
	const outputTokens = integer(data.eval_count);

	if (
		!message &&
		finishReason === undefined &&
		inputTokens === undefined &&
		outputTokens === undefined
	) {
		return null;
	}

	return {
		...(message ?? {}),
		...(finishReason === undefined ? {} : { finishReason }),
		...(inputTokens === undefined ? {} : { inputTokens }),
		...(outputTokens === undefined ? {} : { outputTokens })
	};
}

function integer(value: unknown): number | undefined {
	return typeof value === 'number' && Number.isInteger(value) ? value : undefined;
}
