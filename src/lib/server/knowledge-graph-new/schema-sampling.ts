import type { GraphChunk } from './extraction';

export interface SchemaSampleOptions {
	maxChunks: number;
	maxCharacters: number;
}

export interface SchemaSample {
	chunks: GraphChunk[];
	text: string;
}

export interface SchemaDiscoveryBatchOptions {
	maxBatches: number;
	maxCharactersPerBatch: number;
}

interface IndexedChunk {
	chunk: GraphChunk;
	index: number;
}

export function buildSchemaSample(
	chunks: GraphChunk[],
	options: SchemaSampleOptions
): SchemaSample {
	validateOptions(options);

	const candidates = chunks
		.map((chunk, index) => ({ chunk, index }))
		.filter(({ chunk }) => chunk.content.trim().length > 0);

	if (!candidates.length) {
		return { chunks: [], text: '' };
	}

	const byDocument = new Map<string, IndexedChunk[]>();

	for (const candidate of candidates) {
		const existing = byDocument.get(candidate.chunk.documentId) ?? [];
		existing.push(candidate);
		byDocument.set(candidate.chunk.documentId, existing);
	}

	const documentGroups = selectAcrossBands(
		[...byDocument.values()],
		Math.min(options.maxChunks, byDocument.size)
	);

	const selected: IndexedChunk[] = [];
	const baseAllocation = Math.floor(options.maxChunks / documentGroups.length);
	let remainingAllocation = options.maxChunks % documentGroups.length;

	for (const documentChunks of documentGroups) {
		let allocation = baseAllocation;

		if (remainingAllocation > 0) {
			allocation += 1;
			remainingAllocation -= 1;
		}

		selected.push(
			...selectAcrossBands(documentChunks, Math.min(allocation, documentChunks.length))
		);
	}

	// Fill unused slots when one or more documents had fewer chunks than
	// their initial allocation.
	if (selected.length < options.maxChunks) {
		const selectedIndexes = new Set(selected.map((item) => item.index));
		const remaining = candidates.filter((item) => !selectedIndexes.has(item.index));

		selected.push(
			...selectAcrossBands(
				remaining,
				Math.min(options.maxChunks - selected.length, remaining.length)
			)
		);
	}

	selected.sort((left, right) => left.index - right.index);

	const sampledChunks = selected.map((item) => item.chunk);

	return {
		chunks: sampledChunks,
		text: formatSample(sampledChunks, options.maxCharacters)
	};
}

export function buildSchemaDiscoveryBatches(
	chunks: GraphChunk[],
	options: SchemaDiscoveryBatchOptions
): string[] {
	if (!Number.isInteger(options.maxBatches) || options.maxBatches <= 0) {
		throw new Error('Schema discovery batch count must be a positive integer.');
	}
	if (!Number.isInteger(options.maxCharactersPerBatch) || options.maxCharactersPerBatch < 128) {
		throw new Error('Each schema discovery batch must allow at least 128 characters.');
	}

	const usable = chunks.filter((chunk) => chunk.content.trim());
	if (!usable.length) return [];
	const batchCount = Math.min(options.maxBatches, usable.length);
	const groups = Array.from({ length: batchCount }, () => [] as GraphChunk[]);

	// The selected sample is ordered by corpus position. Round-robin assignment
	// spreads documents and sections across the small independent model calls.
	for (let index = 0; index < usable.length; index += 1) {
		groups[index % batchCount].push(usable[index]);
	}

	return groups.map((group) => formatSample(group, options.maxCharactersPerBatch));
}

function selectAcrossBands<T>(items: T[], count: number): T[] {
	if (count <= 0) return [];
	if (count >= items.length) return [...items];

	const selected: T[] = [];

	for (let band = 0; band < count; band += 1) {
		const start = Math.floor((band * items.length) / count);
		const end = Math.floor(((band + 1) * items.length) / count) - 1;
		const index = Math.floor((start + Math.max(start, end)) / 2);

		selected.push(items[index]);
	}

	return selected;
}

function formatSample(chunks: GraphChunk[], maxCharacters: number): string {
	if (!chunks.length) return '';

	const separatorCharacters = Math.max(0, chunks.length - 1) * 2;
	const sectionBudget = Math.floor((maxCharacters - separatorCharacters) / chunks.length);
	const documentLabels = new Map<string, string>();

	return chunks
		.map((chunk, index) => {
			let documentLabel = documentLabels.get(chunk.documentId);
			if (!documentLabel) {
				documentLabel = `d${documentLabels.size + 1}`;
				documentLabels.set(chunk.documentId, documentLabel);
			}
			const header = `[document:${documentLabel} sample:${index + 1}]\n`;

			if (header.length >= sectionBudget) {
				return header.slice(0, sectionBudget);
			}

			return header + balancedExcerpt(chunk.content, sectionBudget - header.length);
		})
		.join('\n\n');
}

function balancedExcerpt(text: string, limit: number): string {
	const cleanText = text.trim();

	if (cleanText.length <= limit) return cleanText;

	const marker = '\n…\n';
	const usableCharacters = limit - marker.length;

	if (usableCharacters <= 0) {
		return cleanText.slice(0, limit);
	}

	const beginningLength = Math.ceil(usableCharacters / 2);
	const endingLength = usableCharacters - beginningLength;

	return cleanText.slice(0, beginningLength) + marker + cleanText.slice(-endingLength);
}

function validateOptions(options: SchemaSampleOptions): void {
	if (!Number.isInteger(options.maxChunks) || options.maxChunks <= 0) {
		throw new Error('Schema sample chunk count must be a positive integer.');
	}

	if (!Number.isInteger(options.maxCharacters) || options.maxCharacters < options.maxChunks * 128) {
		throw new Error('Schema sample must provide at least 128 characters per chunk.');
	}
}
