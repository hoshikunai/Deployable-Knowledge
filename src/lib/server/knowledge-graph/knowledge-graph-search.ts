// Knowledge-graph search combines hybrid chunks with exact and fuzzy entity-label seeds,
// then adds LightRAG neighborhoods and PathRAG relational traversal before reranking.

import { searchHybrid } from '$lib/server/rag/search/hybrid-search';
import type { SearchChunkType } from '$lib/server/rag/search/search-shared';
import { ensureKnowledgeGraph } from './graph-index';
import { lightRagSearch } from './light-rag';
import { pathRagSearch } from './path-rag';
import { selectGraphSeedCandidates } from './seed-selection';
import type {
	KnowledgeGraphMatch,
	KnowledgeGraphPath,
	KnowledgeGraphSearchResult,
	RelationType
} from './types';
import { personalizedPageRank, type PersonalizedPageRankEvidence } from './personalized-page-rank';
import { unique } from './utils';

export type KnowledgeGraphSearchOptions = {
	query: string;
	topK?: number;
	documentIds?: string[];
	chunkTypes?: SearchChunkType[];
	maxDepth?: number;
};

type ScoreAccumulator = {
	hybridScore: number;
	lightScore: number;
	pathScore: number;
	pprScore: number;
	matchedEntities: string[];
	relations: RelationType[];
	pathCount: number;
};

export type NormalizedKnowledgeGraphScores = {
	hybrid: number;
	light: number;
	path: number;
	ppr: number;
};

export async function searchKnowledgeGraph(
	options: KnowledgeGraphSearchOptions
): Promise<KnowledgeGraphSearchResult> {
	const query = options.query.trim();
	const topK = Math.max(0, Math.floor(options.topK ?? 5));
	if (!query || topK === 0) return { query, results: [], paths: [] };

	// Build a missing or stale selected-document graph automatically. A current graph
	// is reused, and concurrent answer/visualization requests share the same build.
	// Hybrid chunks remain strong grounded seeds; label matching adds graph-native recall.
	const index = await ensureKnowledgeGraph(options.documentIds);
	const hybrid = await searchHybrid({
		query,
		topK: Math.max(topK * 3, 15),
		documentIds: options.documentIds,
		chunkTypes: options.chunkTypes
	});
	const hybridSeeds = hybrid.results.map((match, index) => ({
		...match,
		score: 1 / (index + 1)
	}));
	// The built graph is immutable during retrieval. Query-specific graph cloning and
	// extraction caused the largest query-time memory spike; hybrid chunks and exact or
	// fuzzy entity labels already provide personalized restart seeds.
	const graph = index.graph;
	const seeds = selectGraphSeedCandidates({
		query,
		graph,
		hybridResults: hybridSeeds
	});

	const lightEvidence = lightRagSearch(graph, seeds);

	const pprEvidence = personalizedPageRank(index.pprIndex, seeds, {
		damping: 0.5,
		maxIterations: 50,
		tolerance: 1e-7,
		resultLimit: Math.max(topK * 10, 50)
	});

	const paths = pathRagSearch(
		query,
		graph,
		seeds,
		Math.max(1, Math.min(4, options.maxDepth ?? 3)),
		Math.max(topK * 3, 12)
	);
	const scores = collectScores(hybridSeeds, lightEvidence, paths, pprEvidence);

	const maxHybrid = maxScore([...scores.values()].map((score) => score.hybridScore));

	const maxLight = maxScore([...scores.values()].map((score) => score.lightScore));

	const maxPath = maxScore([...scores.values()].map((score) => score.pathScore));

	const maxPpr = Math.max(
		Number.EPSILON,
		...[...scores.values()].map((score) => score.pprScore).filter(Number.isFinite)
	);

	const allowedTypes = new Set(options.chunkTypes ?? []);

	const results: KnowledgeGraphMatch[] = [];

	for (const [chunkId, score] of scores) {
		const chunk = index.chunksById.get(chunkId);
		if (!chunk) continue;
		if (allowedTypes.size && !allowedTypes.has(chunk.chunkType)) continue;

		// Weighted fusion now lets graph structure lead while hybrid remains a grounding
		// signal. This keeps KG search meaningfully different from plain hybrid search.
		const hybridPart = score.hybridScore / maxHybrid;
		const lightPart = score.lightScore / maxLight;
		const pathPart = score.pathScore / maxPath;
		const pprPart = score.pprScore / maxPpr;
		const graphScore = lightPart * 0.25 + pathPart * 0.25 + pprPart * 0.5;
		const retrievalScore = fuseKnowledgeGraphScores({
			hybrid: hybridPart,
			light: lightPart,
			path: pathPart,
			ppr: pprPart
		});

		results.push({
			...chunk,
			score: clamp01(retrievalScore + acronymDefinitionBoost(query, chunk.content)),
			graphScore: clamp01(graphScore),
			hybridScore: score.hybridScore || undefined,
			matchedEntities: unique(score.matchedEntities),
			relations: unique(score.relations),
			pathCount: score.pathCount
		});
	}

	results.sort((left, right) => right.score - left.score);
	return { query, results: results.slice(0, topK), paths };
}

export function acronymDefinitionBoost(query: string, content: string): number {
	const match = query.match(
		/\b(?:what\s+does|define)\s+([A-Z][A-Z0-9/-]{1,12})\s+(?:stand\s+for|mean)\b/i
	);
	const acronym = match?.[1]?.toUpperCase();
	if (!acronym) return 0;

	const normalized = content.replace(/\s+/g, ' ');
	const escaped = acronym.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
	const definesAcronym =
		new RegExp(`\\b${escaped}\\b\\s*\\(`).test(normalized) ||
		new RegExp(`\\b${escaped}\\b\\s+(?:stands\\s+for|means)\\b`, 'i').test(normalized) ||
		new RegExp(`\\b${escaped}\\b\\s+acronym\\b`, 'i').test(normalized) ||
		new RegExp(`\\bacronym\\s+${escaped}\\b`, 'i').test(normalized);

	// Definition questions need the defining graph evidence to outrank incidental
	// lexical matches such as calendar dates returned by the hybrid seed search.
	return definesAcronym ? 1 : 0;
}

export function fuseKnowledgeGraphScores(scores: NormalizedKnowledgeGraphScores): number {
	return scores.hybrid * 0.4 + scores.light * 0.2 + scores.path * 0.15 + scores.ppr * 0.25;
}

function collectScores(
	hybrid: Array<{ chunkId: string; score: number }>,
	light: Array<{
		chunkId: string;
		score: number;
		matchedEntities: string[];
		relations: RelationType[];
	}>,
	paths: KnowledgeGraphPath[],
	ppr: PersonalizedPageRankEvidence[]
): Map<string, ScoreAccumulator> {
	const scores = new Map<string, ScoreAccumulator>();

	for (const match of hybrid) {
		getScore(scores, match.chunkId).hybridScore = Math.max(0, match.score);
	}

	for (const evidence of light) {
		const score = getScore(scores, evidence.chunkId);
		score.lightScore += evidence.score;
		score.matchedEntities.push(...evidence.matchedEntities);
		score.relations.push(...evidence.relations);
	}

	for (const path of paths) {
		for (const chunkId of path.chunkIds) {
			const score = getScore(scores, chunkId);
			score.pathScore += path.score;
			score.pathCount += 1;
			score.relations.push(...path.edges.map((edge) => edge.relation));
			score.matchedEntities.push(
				...path.nodes.filter((node) => node.kind === 'entity').map((node) => node.label)
			);
		}
	}

	for (const evidence of ppr) {
		const score = getScore(scores, evidence.chunkId);
		score.pprScore = Math.max(score.pprScore, evidence.score);
	}

	return scores;
}

function getScore(scores: Map<string, ScoreAccumulator>, chunkId: string): ScoreAccumulator {
	const existing = scores.get(chunkId);
	if (existing) return existing;

	const created: ScoreAccumulator = {
		hybridScore: 0,
		lightScore: 0,
		pathScore: 0,
		pprScore: 0,
		matchedEntities: [],
		relations: [],
		pathCount: 0
	};
	scores.set(chunkId, created);
	return created;
}

function maxScore(values: number[]): number {
	return Math.max(1, ...values.filter(Number.isFinite));
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
