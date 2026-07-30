import { and, eq, inArray } from 'drizzle-orm';
import type { SQL } from 'drizzle-orm';
import { DEFAULT_ASSISTANT_CONFIG } from '$lib/constants';
import { db } from '../../database/database';
import {
	documentChunks,
	documents,
	hippoChunkEntities,
	hippoChunkFacts,
	hippoFacts,
	hippoIndexMetadata,
	hippoSynonyms
} from '../../database/schema';
import { getProvider } from '../../providers/registry';
import { embedTexts } from '../embedding-model';
import { embeddingDotProduct, embeddingFromBytes } from '../embedding-vectors';
import {
	cleanFilterValues,
	type SearchMatchBase,
	type SearchOptionsBase,
	type SearchResult
} from '../search/search-shared';
import { searchSemantic } from '../search/semantic-search';
import {
	HIPPORAG_DENSE_PASSAGE_WEIGHT,
	HIPPORAG_INDEX_ID,
	HIPPORAG_PPR_DAMPING,
	HIPPORAG_PPR_ITERATIONS,
	HIPPORAG_PPR_TOLERANCE,
	HIPPORAG_QUERY_FACT_COUNT
} from './constants';
import { recognizeFacts } from './openie';

type PassageRow = SearchMatchBase & {
	embedding: Uint8Array | ArrayBuffer | null;
};

type WeightedEdge = { target: string; weight: number };

function entityNode(id: string): string {
	return `entity:${id}`;
}

function passageNode(id: string): string {
	return `passage:${id}`;
}

function addEdge(
	graph: Map<string, Map<string, number>>,
	left: string,
	right: string,
	weight: number
) {
	if (left === right || weight <= 0) return;
	const leftEdges = graph.get(left) ?? new Map<string, number>();
	const rightEdges = graph.get(right) ?? new Map<string, number>();
	leftEdges.set(right, (leftEdges.get(right) ?? 0) + weight);
	rightEdges.set(left, (rightEdges.get(left) ?? 0) + weight);
	graph.set(left, leftEdges);
	graph.set(right, rightEdges);
}

function normalizeSeeds(seeds: Map<string, number>): Map<string, number> {
	const positive = [...seeds].filter(([, weight]) => weight > 0);
	const total = positive.reduce((sum, [, weight]) => sum + weight, 0);
	if (total === 0) return new Map();
	return new Map(positive.map(([node, weight]) => [node, weight / total]));
}

function personalizedPageRank(
	graph: Map<string, Map<string, number>>,
	seedWeights: Map<string, number>
): Map<string, number> {
	const teleport = normalizeSeeds(seedWeights);
	if (teleport.size === 0) return new Map();

	const nodes = new Set<string>([...graph.keys(), ...teleport.keys()]);
	const adjacency = new Map<string, WeightedEdge[]>();
	for (const node of nodes) {
		adjacency.set(
			node,
			[...(graph.get(node) ?? [])].map(([target, weight]) => ({ target, weight }))
		);
	}

	let scores = new Map<string, number>(teleport);
	for (let iteration = 0; iteration < HIPPORAG_PPR_ITERATIONS; iteration += 1) {
		const next = new Map<string, number>();
		for (const [node, probability] of teleport) {
			next.set(node, HIPPORAG_PPR_DAMPING * probability);
		}

		let danglingMass = 0;
		for (const node of nodes) {
			const score = scores.get(node) ?? 0;
			const edges = adjacency.get(node) ?? [];
			const totalWeight = edges.reduce((sum, edge) => sum + edge.weight, 0);
			if (totalWeight === 0) {
				danglingMass += score;
				continue;
			}
			for (const edge of edges) {
				const contribution = (1 - HIPPORAG_PPR_DAMPING) * score * (edge.weight / totalWeight);
				next.set(edge.target, (next.get(edge.target) ?? 0) + contribution);
			}
		}

		if (danglingMass > 0) {
			for (const [node, probability] of teleport) {
				next.set(
					node,
					(next.get(node) ?? 0) + (1 - HIPPORAG_PPR_DAMPING) * danglingMass * probability
				);
			}
		}

		let delta = 0;
		for (const node of nodes) delta += Math.abs((next.get(node) ?? 0) - (scores.get(node) ?? 0));
		scores = next;
		if (delta < HIPPORAG_PPR_TOLERANCE) break;
	}

	return scores;
}

async function denseFallback(options: SearchOptionsBase): Promise<SearchResult<SearchMatchBase>> {
	const result = await searchSemantic(options);
	return {
		query: result.query,
		results: result.results.map(({ score: _score, ...match }) => match)
	};
}

async function loadPassages(options: SearchOptionsBase): Promise<PassageRow[]> {
	const filters: SQL[] = [eq(documents.active, true)];
	const documentIds = cleanFilterValues(options.documentIds);
	const sourcePaths = cleanFilterValues(options.sourcePaths);
	const chunkTypes = cleanFilterValues(options.chunkTypes);

	if (documentIds.length > 0) filters.push(inArray(documentChunks.documentId, documentIds));
	if (sourcePaths.length > 0) filters.push(inArray(documents.sourcePath, sourcePaths));
	if (chunkTypes.length > 0) filters.push(inArray(documentChunks.chunkType, chunkTypes));

	return (await db
		.select({
			chunkId: documentChunks.id,
			documentId: documentChunks.documentId,
			sourcePath: documents.sourcePath,
			sourceTitle: documents.title,
			sourceType: documents.sourceType,
			pageIndex: documentChunks.pageIndex,
			chunkIndex: documentChunks.chunkIndex,
			chunkType: documentChunks.chunkType,
			content: documentChunks.content,
			embedding: documentChunks.embedding
		})
		.from(documentChunks)
		.innerJoin(documents, eq(documents.id, documentChunks.documentId))
		.where(and(...filters))) as PassageRow[];
}

export async function searchHippoRag2(
	options: SearchOptionsBase
): Promise<SearchResult<SearchMatchBase>> {
	const query = options.query.trim();
	const topK = Math.max(0, Math.floor(options.topK ?? DEFAULT_ASSISTANT_CONFIG.ragTopK));
	if (!query || topK === 0) return { query, results: [] };

	const metadata = await db
		.select()
		.from(hippoIndexMetadata)
		.where(eq(hippoIndexMetadata.id, HIPPORAG_INDEX_ID))
		.get();
	if (!metadata) return denseFallback({ ...options, query, topK });

	const passages = await loadPassages(options);
	if (passages.length === 0) return { query, results: [] };
	if (passages.some(({ embedding }) => !embedding)) {
		return denseFallback({ ...options, query, topK });
	}

	const passageIds = passages.map(({ chunkId }) => chunkId);
	const factRows = await db
		.selectDistinct({
			id: hippoFacts.id,
			subjectEntityId: hippoFacts.subjectEntityId,
			objectEntityId: hippoFacts.objectEntityId,
			content: hippoFacts.content,
			embedding: hippoFacts.embedding
		})
		.from(hippoFacts)
		.innerJoin(hippoChunkFacts, eq(hippoChunkFacts.factId, hippoFacts.id))
		.where(inArray(hippoChunkFacts.chunkId, passageIds));
	if (factRows.length === 0) return denseFallback({ ...options, query, topK });

	const queryEmbedding = (await embedTexts([query], 'search_query'))[0] ?? [];
	const rankedFacts = factRows
		.map((fact) => ({
			...fact,
			score: embeddingDotProduct(queryEmbedding, embeddingFromBytes(fact.embedding))
		}))
		.sort((left, right) => right.score - left.score)
		.slice(0, HIPPORAG_QUERY_FACT_COUNT);

	let recognizedIndices: number[];
	try {
		const provider = getProvider(metadata.providerId);
		recognizedIndices = await recognizeFacts(
			provider,
			metadata.modelId,
			query,
			rankedFacts.map(({ content }) => content)
		);
	} catch {
		return denseFallback({ ...options, query, topK });
	}
	if (recognizedIndices.length === 0) return denseFallback({ ...options, query, topK });

	const entityLinks = await db
		.select({ chunkId: hippoChunkEntities.chunkId, entityId: hippoChunkEntities.entityId })
		.from(hippoChunkEntities)
		.where(inArray(hippoChunkEntities.chunkId, passageIds));
	const entityIds = [...new Set(entityLinks.map(({ entityId }) => entityId))];
	const [relations, synonyms] = await Promise.all([
		db
			.select({
				subjectEntityId: hippoFacts.subjectEntityId,
				objectEntityId: hippoFacts.objectEntityId
			})
			.from(hippoFacts)
			.where(
				inArray(
					hippoFacts.id,
					factRows.map(({ id }) => id)
				)
			),
		entityIds.length === 0
			? Promise.resolve([])
			: db
					.select({
						sourceEntityId: hippoSynonyms.sourceEntityId,
						targetEntityId: hippoSynonyms.targetEntityId,
						score: hippoSynonyms.score
					})
					.from(hippoSynonyms)
					.where(inArray(hippoSynonyms.sourceEntityId, entityIds))
	]);

	const allowedEntities = new Set(entityIds);
	const graph = new Map<string, Map<string, number>>();
	for (const link of entityLinks) {
		addEdge(graph, entityNode(link.entityId), passageNode(link.chunkId), 1);
	}
	for (const relation of relations) {
		if (
			allowedEntities.has(relation.subjectEntityId) &&
			allowedEntities.has(relation.objectEntityId)
		) {
			addEdge(graph, entityNode(relation.subjectEntityId), entityNode(relation.objectEntityId), 1);
		}
	}
	for (const synonym of synonyms) {
		if (allowedEntities.has(synonym.targetEntityId)) {
			addEdge(
				graph,
				entityNode(synonym.sourceEntityId),
				entityNode(synonym.targetEntityId),
				synonym.score
			);
		}
	}

	const seeds = new Map<string, number>();
	for (const index of recognizedIndices) {
		const fact = rankedFacts[index];
		const weight = Math.max(Number.EPSILON, fact.score);
		seeds.set(
			entityNode(fact.subjectEntityId),
			Math.max(seeds.get(entityNode(fact.subjectEntityId)) ?? 0, weight)
		);
		seeds.set(
			entityNode(fact.objectEntityId),
			Math.max(seeds.get(entityNode(fact.objectEntityId)) ?? 0, weight)
		);
	}
	for (const passage of passages) {
		const score = embeddingDotProduct(queryEmbedding, embeddingFromBytes(passage.embedding!));
		seeds.set(passageNode(passage.chunkId), Math.max(0, score) * HIPPORAG_DENSE_PASSAGE_WEIGHT);
	}

	const scores = personalizedPageRank(graph, seeds);
	const byChunkId = new Map(passages.map((passage) => [passage.chunkId, passage]));
	const results = [...byChunkId.values()]
		.map(({ embedding: _embedding, ...passage }) => ({
			passage,
			score: scores.get(passageNode(passage.chunkId)) ?? 0
		}))
		.sort((left, right) => right.score - left.score)
		.slice(0, topK)
		.map(({ passage }) => passage);

	return { query, results };
}
