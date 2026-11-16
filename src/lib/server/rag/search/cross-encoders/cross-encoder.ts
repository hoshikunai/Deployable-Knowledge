import type { DownloadProgressHandler } from '#lib/server/utils/download-progress';

export interface CrossEncoder {
	readonly id: string;
	readonly name: string;
	readonly maxSupportedTokens: number;

	isInstalled(): Promise<boolean>;
	prepare(onProgress: DownloadProgressHandler): Promise<void>;
	predict(
		query: string,
		passages: readonly string[],
		maxTokens?: number
	): Promise<readonly number[]>;
}

export function sigmoidScore(value: number): number {
	return 1 / (1 + Math.exp(-value));
}
