import { createHash, randomUUID } from 'node:crypto';
import { count, eq, isNull, notExists } from 'drizzle-orm';
import { EMBEDDING_MODEL, embedTexts } from '../embedding-model';
import { embeddingDotProduct, embeddingFromBytes, embeddingToBuffer } from '../embedding-vectors';
import { db } from '../../database/database';
import {
	documentChunks,
	hippoChunkEntities,
	hippoChunkFacts,
	hippoChunkIndex,
	hippoEntities,
	hippoFacts,
	hippoIndexMetadata,
	hippoSynonyms
} from '../../database/schema';
import { getProvider } from '../../providers/registry';
import type { ApiHippoRagIndexStatus } from '$lib/types';
import {
	HIPPORAG_INDEX_ID,
	HIPPORAG_PROMPT_VERSION,
	HIPPORAG_SYNONYM_THRESHOLD,
	HIPPORAG_SYNONYM_TOP_K
} from './constants';
import { extractOpenIe, normalizeEntity } from './openie';
import type { HippoIndexConfig, HippoIndexProgress, HippoTriple } from './types';

const INSERT_BATCH_SIZE = 100;

let running = false;
let lastError: string | null = null;

function stableId(namespace: string, value: string): string {
	return createHash('sha256').update(`${namespace}:${value}`).digest('hex');
}

function entityId(name: string): string {
	return stableId('entity', normalizeEntity(name));
}

function factContent(triple: HippoTriple): string {
	return `${triple.subject} ${triple.predicate} ${triple.object}`;
}

function factId(triple: HippoTriple): string {
	const key = [
		normalizeEntity(triple.subject),
		triple.predicate.replace(/\s+/g, ' ').trim().toLocaleLowerCase(),
		normalizeEntity(triple.object)
	].join('\u0000');
	return stableId('fact', key);
}

async function clearIndex(): Promise<void> {
	await db.transaction(async (transaction) => {
		await transaction.delete(hippoChunkIndex);
		await transaction.delete(hippoChunkFacts);
		await transaction.delete(hippoChunkEntities);
		await transaction.delete(hippoSynonyms);
		await transaction.delete(hippoFacts);
		await transaction.delete(hippoEntities);
		await transaction.delete(hippoIndexMetadata);
	});
}

async function insertInBatches<T>(
	rows: readonly T[],
	insert: (batch: T[]) => Promise<unknown>
): Promise<void> {
	for (let index = 0; index < rows.length; index += INSERT_BATCH_SIZE) {
		await insert(rows.slice(index, index + INSERT_BATCH_SIZE));
	}
}

async function indexChunk(
	chunk: { id: string; content: string },
	config: HippoIndexConfig,
	signal?: AbortSignal
): Promise<void> {
	signal?.throwIfAborted();
	const provider = getProvider(config.providerId);
	const openIe = await extractOpenIe(provider, config.modelId, chunk.content, signal);
	const entityNames = new Map<string, string>();

	for (const name of openIe.entities) entityNames.set(entityId(name), name);
	for (const triple of openIe.triples) {
		entityNames.set(entityId(triple.subject), triple.subject);
		entityNames.set(entityId(triple.object), triple.object);
	}

	const entityRows = [...entityNames].map(([id, name]) => ({ id, name }));
	const entityEmbeddings = await embedTexts(
		entityRows.map(({ name }) => name),
		'search_document'
	);
	const factRows = openIe.triples.map((triple) => ({
		id: factId(triple),
		subjectEntityId: entityId(triple.subject),
		predicate: triple.predicate,
		objectEntityId: entityId(triple.object),
		content: factContent(triple)
	}));
	const factEmbeddings = await embedTexts(
		factRows.map(({ content }) => content),
		'search_document'
	);
	const now = new Date().toISOString();

	await db.transaction(async (transaction) => {
		if (entityRows.length > 0) {
			await transaction
				.insert(hippoEntities)
				.values(
					entityRows.map((row, index) => ({
						...row,
						embedding: embeddingToBuffer(entityEmbeddings[index] ?? []),
						createdAt: now
					}))
				)
				.onConflictDoNothing();
			await transaction
				.insert(hippoChunkEntities)
				.values(
					entityRows.map(({ id }) => ({
						id: randomUUID(),
						chunkId: chunk.id,
						entityId: id
					}))
				)
				.onConflictDoNothing();
		}

		if (factRows.length > 0) {
			await transaction
				.insert(hippoFacts)
				.values(
					factRows.map((row, index) => ({
						...row,
						embedding: embeddingToBuffer(factEmbeddings[index] ?? []),
						createdAt: now
					}))
				)
				.onConflictDoNothing();
			await transaction
				.insert(hippoChunkFacts)
				.values(
					factRows.map(({ id }) => ({
						id: randomUUID(),
						chunkId: chunk.id,
						factId: id
					}))
				)
				.onConflictDoNothing();
		}

		await transaction
			.insert(hippoChunkIndex)
			.values({ chunkId: chunk.id, indexedAt: now })
			.onConflictDoUpdate({
				target: hippoChunkIndex.chunkId,
				set: { indexedAt: now }
			});
	});
}

async function rebuildSynonymEdges(onProgress?: (progress: HippoIndexProgress) => void) {
	await db
		.delete(hippoFacts)
		.where(
			notExists(
				db
					.select({ id: hippoChunkFacts.id })
					.from(hippoChunkFacts)
					.where(eq(hippoChunkFacts.factId, hippoFacts.id))
			)
		);
	await db
		.delete(hippoEntities)
		.where(
			notExists(
				db
					.select({ id: hippoChunkEntities.id })
					.from(hippoChunkEntities)
					.where(eq(hippoChunkEntities.entityId, hippoEntities.id))
			)
		);

	const rows = await db
		.select({ id: hippoEntities.id, embedding: hippoEntities.embedding })
		.from(hippoEntities);
	const entities = rows.map((row) => ({
		id: row.id,
		vector: embeddingFromBytes(row.embedding)
	}));
	const edges: Array<{
		id: string;
		sourceEntityId: string;
		targetEntityId: string;
		score: number;
	}> = [];

	for (let leftIndex = 0; leftIndex < entities.length; leftIndex += 1) {
		const left = entities[leftIndex];
		const neighbors: Array<{ id: string; score: number }> = [];

		for (let rightIndex = 0; rightIndex < entities.length; rightIndex += 1) {
			if (leftIndex === rightIndex) continue;
			const right = entities[rightIndex];
			const score = embeddingDotProduct(left.vector, right.vector);
			if (score >= HIPPORAG_SYNONYM_THRESHOLD) neighbors.push({ id: right.id, score });
		}

		neighbors.sort((a, b) => b.score - a.score);
		for (const neighbor of neighbors.slice(0, HIPPORAG_SYNONYM_TOP_K)) {
			const [sourceEntityId, targetEntityId] = [left.id, neighbor.id].sort();
			edges.push({
				id: stableId('synonym', `${sourceEntityId}:${targetEntityId}`),
				sourceEntityId,
				targetEntityId,
				score: neighbor.score
			});
		}

		onProgress?.({
			stage: 'linking',
			current: leftIndex + 1,
			total: entities.length,
			message: `Linking entity ${leftIndex + 1} of ${entities.length}`
		});
	}

	await db.delete(hippoSynonyms);
	const uniqueEdges = [...new Map(edges.map((edge) => [edge.id, edge])).values()];
	await insertInBatches(uniqueEdges, (batch) =>
		db.insert(hippoSynonyms).values(batch).onConflictDoNothing()
	);
}

export async function getHippoIndexStatus(): Promise<ApiHippoRagIndexStatus> {
	const [chunkCount] = await db.select({ value: count() }).from(documentChunks);
	const [indexedCount] = await db.select({ value: count() }).from(hippoChunkIndex);
	const metadata = await db
		.select()
		.from(hippoIndexMetadata)
		.where(eq(hippoIndexMetadata.id, HIPPORAG_INDEX_ID))
		.get();
	const totalChunks = chunkCount?.value ?? 0;
	const indexedChunks = indexedCount?.value ?? 0;

	return {
		error: lastError,
		indexedChunks,
		modelId: metadata?.modelId ?? null,
		pendingChunks: Math.max(0, totalChunks - indexedChunks),
		providerId: metadata?.providerId ?? null,
		ready: totalChunks > 0 && totalChunks === indexedChunks,
		running,
		totalChunks
	};
}

export async function buildHippoIndex(
	config: HippoIndexConfig,
	rebuild: boolean,
	onProgress?: (progress: HippoIndexProgress) => void,
	signal?: AbortSignal
): Promise<ApiHippoRagIndexStatus> {
	if (running) throw new Error('A HippoRAG2 index build is already running.');
	running = true;
	lastError = null;

	try {
		const metadata = await db
			.select()
			.from(hippoIndexMetadata)
			.where(eq(hippoIndexMetadata.id, HIPPORAG_INDEX_ID))
			.get();
		const configChanged =
			metadata &&
			(metadata.providerId !== config.providerId ||
				metadata.modelId !== config.modelId ||
				metadata.embeddingModel !== EMBEDDING_MODEL ||
				metadata.promptVersion !== HIPPORAG_PROMPT_VERSION);

		if (rebuild || configChanged) await clearIndex();

		const now = new Date().toISOString();
		await db
			.insert(hippoIndexMetadata)
			.values({
				id: HIPPORAG_INDEX_ID,
				providerId: config.providerId,
				modelId: config.modelId,
				embeddingModel: EMBEDDING_MODEL,
				promptVersion: HIPPORAG_PROMPT_VERSION,
				updatedAt: now
			})
			.onConflictDoUpdate({
				target: hippoIndexMetadata.id,
				set: {
					...config,
					embeddingModel: EMBEDDING_MODEL,
					promptVersion: HIPPORAG_PROMPT_VERSION,
					updatedAt: now
				}
			});

		const chunks = await db
			.select({ id: documentChunks.id, content: documentChunks.content })
			.from(documentChunks)
			.leftJoin(hippoChunkIndex, eq(hippoChunkIndex.chunkId, documentChunks.id))
			.where(isNull(hippoChunkIndex.chunkId));

		for (let index = 0; index < chunks.length; index += 1) {
			signal?.throwIfAborted();
			onProgress?.({
				stage: 'extracting',
				current: index,
				total: chunks.length,
				message: `Extracting chunk ${index + 1} of ${chunks.length}`
			});
			await indexChunk(chunks[index], config, signal);
			onProgress?.({
				stage: 'embedding',
				current: index + 1,
				total: chunks.length,
				message: `Indexed chunk ${index + 1} of ${chunks.length}`
			});
		}

		await rebuildSynonymEdges(onProgress);
		onProgress?.({
			stage: 'finalizing',
			current: 1,
			total: 1,
			message: 'HippoRAG2 index ready'
		});
		running = false;
		return await getHippoIndexStatus();
	} catch (error) {
		lastError = error instanceof Error ? error.message : String(error);
		throw error;
	} finally {
		running = false;
	}
}
