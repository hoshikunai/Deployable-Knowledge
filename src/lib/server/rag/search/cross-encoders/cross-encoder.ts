export interface CrossEncoder {
	readonly id: string;
	readonly name: string;
	// Longest query + passage input, in tokens, the model's position embeddings allow
	readonly maxSupportedTokens: number;

	// Inputs longer than maxTokens are truncated before scoring
	predict(
		query: string,
		passages: readonly string[],
		maxTokens: number
	): Promise<readonly number[]>;
}

export function sigmoidScore(value: number): number {
	return 1 / (1 + Math.exp(-value));
}
