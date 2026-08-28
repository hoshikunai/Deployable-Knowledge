import { isSemanticRelationCategory } from './assertion-quality';
import type { CorpusSchema, SchemaCategory } from './extraction';

export interface SchemaDiscoveryProposal {
	entityTypes: SchemaCategory[];
	relationTypes: SchemaCategory[];
}

export interface SchemaQualityOptions {
	discoveryProposals: SchemaDiscoveryProposal[];
	entityConsolidation: 'complete' | 'failed';
	relationConsolidation: 'complete' | 'chunked' | 'failed';
	minimumScore: number;
}

export interface SchemaQualityMetrics {
	structuralClosure: number;
	semanticRelations: number;
	boundedEndpoints: number;
	distinctRelations: number;
	candidateCoverage: number;
	discoveryBatchCoverage: number;
	consolidationReliability: number;
}

export interface SchemaQualityIssue {
	severity: 'error' | 'warning';
	code: string;
	message: string;
}

export interface SchemaQualityReport {
	status: 'passed' | 'failed';
	score: number;
	minimumScore: number;
	metrics: SchemaQualityMetrics;
	issues: SchemaQualityIssue[];
}

const METRIC_WEIGHTS: Record<keyof SchemaQualityMetrics, number> = {
	structuralClosure: 0.2,
	semanticRelations: 0.15,
	boundedEndpoints: 0.1,
	distinctRelations: 0.1,
	candidateCoverage: 0.2,
	discoveryBatchCoverage: 0.15,
	consolidationReliability: 0.1
};

/**
 * Score an adaptive schema without relying on corpus-specific terms.
 *
 * Closure and semantic relation validity are hard requirements. The remaining
 * metrics make consolidation loss, broad endpoint unions, duplicate predicates,
 * and discovery-batch collapse visible and collectively fail a weak schema.
 */
export function evaluateSchemaQuality(
	schema: CorpusSchema,
	options: SchemaQualityOptions
): SchemaQualityReport {
	validateOptions(options);

	const relationTypes = schema.relationTypes;
	const entityNames = new Set(schema.entityTypes.map((type) => normalize(type.name)));
	const structuralClosure = ratio(
		relationTypes.filter(
			(relation) =>
				Boolean(relation.subjectTypes?.length) &&
				Boolean(relation.objectTypes?.length) &&
				[...(relation.subjectTypes ?? []), ...(relation.objectTypes ?? [])].every((type) =>
					entityNames.has(normalize(type))
				)
		).length,
		relationTypes.length
	);
	const semanticRelations = ratio(
		relationTypes.filter(isSemanticRelationCategory).length,
		relationTypes.length
	);
	const endpointLimit = Math.max(3, Math.ceil(Math.sqrt(schema.entityTypes.length)));
	const boundedEndpoints = ratio(
		relationTypes.filter(
			(relation) =>
				(relation.subjectTypes?.length ?? 0) <= endpointLimit &&
				(relation.objectTypes?.length ?? 0) <= endpointLimit
		).length,
		relationTypes.length
	);
	const distinctRelations = ratio(relationClusters(relationTypes).length, relationTypes.length);

	const semanticCandidates = options.discoveryProposals.flatMap((proposal) =>
		proposal.relationTypes.filter(isSemanticRelationCategory)
	);
	const candidateClusters = relationClusters(semanticCandidates);
	const coveredCandidateClusters = candidateClusters.filter((cluster) =>
		cluster.some((candidate) =>
			relationTypes.some((relation) => relationsMatch(relation, candidate))
		)
	).length;
	const candidateCoverage = ratio(coveredCandidateClusters, candidateClusters.length);
	const discoveryBatchCoverage = ratio(
		options.discoveryProposals.filter((proposal) => {
			const candidates = proposal.relationTypes.filter(isSemanticRelationCategory);
			return candidates.some((candidate) =>
				relationTypes.some((relation) => relationsMatch(relation, candidate))
			);
		}).length,
		options.discoveryProposals.length
	);
	const consolidationReliability = consolidationScore(options);
	const metrics: SchemaQualityMetrics = {
		structuralClosure,
		semanticRelations,
		boundedEndpoints,
		distinctRelations,
		candidateCoverage,
		discoveryBatchCoverage,
		consolidationReliability
	};
	const score = round(
		(Object.keys(METRIC_WEIGHTS) as Array<keyof SchemaQualityMetrics>).reduce(
			(total, metric) => total + metrics[metric] * METRIC_WEIGHTS[metric],
			0
		)
	);
	const issues = qualityIssues(schema, metrics, options.minimumScore, score, endpointLimit);

	return {
		status: issues.some((issue) => issue.severity === 'error') ? 'failed' : 'passed',
		score,
		minimumScore: options.minimumScore,
		metrics: {
			structuralClosure: round(metrics.structuralClosure),
			semanticRelations: round(metrics.semanticRelations),
			boundedEndpoints: round(metrics.boundedEndpoints),
			distinctRelations: round(metrics.distinctRelations),
			candidateCoverage: round(metrics.candidateCoverage),
			discoveryBatchCoverage: round(metrics.discoveryBatchCoverage),
			consolidationReliability: round(metrics.consolidationReliability)
		},
		issues
	};
}

export function schemaQualityFailureMessage(report: SchemaQualityReport): string {
	const diagnostics = report.issues.map((issue) => `${issue.code}: ${issue.message}`).join(' ');
	const reason =
		report.score < report.minimumScore
			? `${report.score.toFixed(3)} < ${report.minimumScore.toFixed(3)}`
			: `hard requirement failed at score ${report.score.toFixed(3)}`;
	return `Schema quality gate failed (${reason}). ${diagnostics}`;
}

function qualityIssues(
	schema: CorpusSchema,
	metrics: SchemaQualityMetrics,
	minimumScore: number,
	score: number,
	endpointLimit: number
): SchemaQualityIssue[] {
	const issues: SchemaQualityIssue[] = [];
	if (!schema.relationTypes.length) {
		issues.push({
			severity: 'error',
			code: 'no-relations',
			message: 'The finalized schema contains no relation types.'
		});
	}
	if (metrics.structuralClosure < 1) {
		issues.push({
			severity: 'error',
			code: 'schema-not-closed',
			message: 'At least one relation has an empty or missing endpoint entity type.'
		});
	}
	if (metrics.semanticRelations < 1) {
		issues.push({
			severity: 'error',
			code: 'non-semantic-relation',
			message:
				'At least one retained relation represents provenance, co-occurrence, or vague association.'
		});
	}
	if (metrics.boundedEndpoints < 0.8) {
		issues.push({
			severity: 'warning',
			code: 'broad-endpoints',
			message: `Only ${percent(metrics.boundedEndpoints)} of relations stay within ${endpointLimit} subject and object types.`
		});
	}
	if (metrics.distinctRelations < 0.85) {
		issues.push({
			severity: 'warning',
			code: 'redundant-relations',
			message: `Only ${percent(metrics.distinctRelations)} of retained relations are distinct relation families.`
		});
	}
	if (metrics.candidateCoverage < 0.6) {
		issues.push({
			severity: 'warning',
			code: 'low-candidate-coverage',
			message: `Only ${percent(metrics.candidateCoverage)} of distinct semantic relation candidates survived consolidation.`
		});
	}
	if (metrics.discoveryBatchCoverage < 2 / 3) {
		issues.push({
			severity: 'warning',
			code: 'low-batch-coverage',
			message: `Only ${percent(metrics.discoveryBatchCoverage)} of successful discovery batches contributed a retained relation.`
		});
	}
	if (metrics.consolidationReliability < 1) {
		issues.push({
			severity: 'warning',
			code: 'consolidation-fallback',
			message: 'Schema consolidation used a fallback path.'
		});
	}
	if (score < minimumScore) {
		issues.push({
			severity: 'error',
			code: 'score-below-threshold',
			message: `The weighted quality score ${score.toFixed(3)} is below ${minimumScore.toFixed(3)}.`
		});
	}
	return issues;
}

function relationClusters(relations: SchemaCategory[]): SchemaCategory[][] {
	const clusters: SchemaCategory[][] = [];
	for (const relation of relations) {
		const cluster = clusters.find((items) => items.some((item) => relationsMatch(item, relation)));
		if (cluster) cluster.push(relation);
		else clusters.push([relation]);
	}
	return clusters;
}

function relationsMatch(left: SchemaCategory, right: SchemaCategory): boolean {
	if (normalize(left.name) === normalize(right.name)) return true;
	if (!endpointOverlap(left.subjectTypes, right.subjectTypes)) return false;
	if (!endpointOverlap(left.objectTypes, right.objectTypes)) return false;

	const leftName = contentWords(left.name);
	const rightName = contentWords(right.name);
	if (overlapRatio(leftName, rightName) >= 0.75) return true;

	return jaccard(contentWords(left.description), contentWords(right.description)) >= 0.35;
}

function endpointOverlap(left: string[] | undefined, right: string[] | undefined): boolean {
	if (!left?.length || !right?.length) return false;
	const normalized = new Set(left.map(normalize));
	return right.some((type) => normalized.has(normalize(type)));
}

function contentWords(value: string): Set<string> {
	const words = normalize(value).split('_').filter(Boolean);
	return new Set(
		words
			.filter(
				(word) =>
					![
						'a',
						'an',
						'the',
						'is',
						'are',
						'was',
						'were',
						'has',
						'have',
						'for',
						'by',
						'to',
						'of',
						'in',
						'with',
						'from',
						'and',
						'or'
					].includes(word)
			)
			.map(stem)
	);
}

function stem(value: string): string {
	if (value.length > 5 && value.endsWith('ing')) return value.slice(0, -3);
	if (value.length > 4 && value.endsWith('ed')) return value.slice(0, -2);
	if (value.length > 4 && value.endsWith('s')) return value.slice(0, -1);
	return value;
}

function overlapRatio(left: Set<string>, right: Set<string>): number {
	if (!left.size || !right.size) return 0;
	const overlap = [...left].filter((word) => right.has(word)).length;
	return overlap / Math.min(left.size, right.size);
}

function jaccard(left: Set<string>, right: Set<string>): number {
	if (!left.size || !right.size) return 0;
	const intersection = [...left].filter((word) => right.has(word)).length;
	return intersection / new Set([...left, ...right]).size;
}

function consolidationScore(options: SchemaQualityOptions): number {
	const entityScore = options.entityConsolidation === 'complete' ? 1 : 0.5;
	const relationScores = { complete: 1, chunked: 0.75, failed: 0 } as const;
	return (entityScore + relationScores[options.relationConsolidation]) / 2;
}

function normalize(value: string): string {
	return value
		.trim()
		.replace(/([a-z])([A-Z])/g, '$1_$2')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '_')
		.replace(/^_+|_+$/g, '');
}

function ratio(numerator: number, denominator: number): number {
	return denominator === 0 ? 0 : numerator / denominator;
}

function round(value: number): number {
	return Math.round(value * 10_000) / 10_000;
}

function percent(value: number): string {
	return `${(value * 100).toFixed(1)}%`;
}

function validateOptions(options: SchemaQualityOptions): void {
	if (
		!Number.isFinite(options.minimumScore) ||
		options.minimumScore < 0 ||
		options.minimumScore > 1
	) {
		throw new Error('The schema quality threshold must be between 0 and 1.');
	}
	if (!options.discoveryProposals.length) {
		throw new Error('Schema quality evaluation requires at least one discovery proposal.');
	}
}
