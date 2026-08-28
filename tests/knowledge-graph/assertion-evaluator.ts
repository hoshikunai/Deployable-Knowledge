import type { GraphAssertion } from '../../src/lib/server/knowledge-graph-new/knowledge-graph';
import type { GoldAssertion, GoldBenchmark, GoldEntity } from './benchmark-types';

export type EvaluatedGraphAssertion = Pick<
	GraphAssertion,
	| 'id'
	| 'documentId'
	| 'chunkId'
	| 'subject'
	| 'subjectType'
	| 'canonicalPredicate'
	| 'object'
	| 'objectType'
	| 'status'
> & {
	provenance: Pick<GraphAssertion['provenance'], 'modality'>;
};

export interface AssertionMatch {
	actualAssertionId: string;
	goldAssertionId: string;
	required: boolean;
	typesCorrect: boolean;
	statusCorrect: boolean;
	modalityCorrect: boolean;
}

export interface AssertionEvaluation {
	actualAssertionCount: number;
	requiredGoldCount: number;
	optionalGoldCount: number;
	matchedRequiredCount: number;
	matchedOptionalCount: number;
	falsePositiveCount: number;
	missingRequiredCount: number;
	reversedAssertionCount: number;
	precision: number;
	recall: number;
	f1: number;
	directionAccuracy: number;
	endpointTypeAccuracy: number;
	statusAccuracy: number;
	modalityAccuracy: number;
	canonicalRelationCoverage: number;
	matches: AssertionMatch[];
}

export function evaluateAssertions(
	benchmark: GoldBenchmark,
	allActualAssertions: EvaluatedGraphAssertion[]
): AssertionEvaluation {
	const chunkIds = new Set(benchmark.chunks.map((chunk) => chunk.chunkId));
	const actual = allActualAssertions.filter((assertion) => chunkIds.has(assertion.chunkId));
	const matching = maximumMatching(actual, benchmark.assertions);
	const matches = [...matching.entries()].map(([actualIndex, goldIndex]) => {
		const candidate = actual[actualIndex];
		const gold = benchmark.assertions[goldIndex];
		return {
			actualAssertionId: candidate.id,
			goldAssertionId: gold.id,
			required: gold.required,
			typesCorrect:
				normalize(candidate.subjectType) === normalize(gold.subject.type) &&
				normalize(candidate.objectType) === normalize(gold.object.type),
			statusCorrect: candidate.status === gold.status,
			modalityCorrect: candidate.provenance.modality === gold.modality
		};
	});
	const matchedActual = new Set(matching.keys());
	const matchedGold = new Set(matching.values());
	const reversedAssertionCount = actual.filter(
		(candidate, index) =>
			!matchedActual.has(index) &&
			benchmark.assertions.some(
				(gold, goldIndex) => !matchedGold.has(goldIndex) && isReversedMatch(candidate, gold)
			)
	).length;
	const matchedRequiredCount = matches.filter((match) => match.required).length;
	const matchedOptionalCount = matches.length - matchedRequiredCount;
	const requiredGoldCount = benchmark.assertions.filter((assertion) => assertion.required).length;
	const precision = ratio(matches.length, actual.length);
	const recall = ratio(matchedRequiredCount, requiredGoldCount);
	const directionDenominator = matches.length + reversedAssertionCount;

	return {
		actualAssertionCount: actual.length,
		requiredGoldCount,
		optionalGoldCount: benchmark.assertions.length - requiredGoldCount,
		matchedRequiredCount,
		matchedOptionalCount,
		falsePositiveCount: actual.length - matches.length,
		missingRequiredCount: requiredGoldCount - matchedRequiredCount,
		reversedAssertionCount,
		precision,
		recall,
		f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
		directionAccuracy: ratio(matches.length, directionDenominator),
		endpointTypeAccuracy: ratio(
			matches.filter((match) => match.typesCorrect).length,
			matches.length
		),
		statusAccuracy: ratio(matches.filter((match) => match.statusCorrect).length, matches.length),
		modalityAccuracy: ratio(
			matches.filter((match) => match.modalityCorrect).length,
			matches.length
		),
		canonicalRelationCoverage: ratio(
			actual.filter((assertion) =>
				benchmark.canonicalRelations.includes(assertion.canonicalPredicate)
			).length,
			actual.length
		),
		matches
	};
}

function maximumMatching(
	actual: EvaluatedGraphAssertion[],
	gold: GoldAssertion[]
): Map<number, number> {
	const goldToActual = new Map<number, number>();
	for (let actualIndex = 0; actualIndex < actual.length; actualIndex += 1) {
		assign(actualIndex, new Set<number>());
	}

	function assign(actualIndex: number, visited: Set<number>): boolean {
		for (let goldIndex = 0; goldIndex < gold.length; goldIndex += 1) {
			if (visited.has(goldIndex) || !isDirectedMatch(actual[actualIndex], gold[goldIndex]))
				continue;
			visited.add(goldIndex);
			const previousActual = goldToActual.get(goldIndex);
			if (previousActual === undefined || assign(previousActual, visited)) {
				goldToActual.set(goldIndex, actualIndex);
				return true;
			}
		}
		return false;
	}

	return new Map([...goldToActual].map(([goldIndex, actualIndex]) => [actualIndex, goldIndex]));
}

function isDirectedMatch(actual: EvaluatedGraphAssertion, gold: GoldAssertion): boolean {
	return (
		sameProvenance(actual, gold) &&
		actual.canonicalPredicate === gold.predicate &&
		matchesEntity(actual.subject, gold.subject) &&
		matchesEntity(actual.object, gold.object)
	);
}

function isReversedMatch(actual: EvaluatedGraphAssertion, gold: GoldAssertion): boolean {
	return (
		sameProvenance(actual, gold) &&
		actual.canonicalPredicate === gold.predicate &&
		matchesEntity(actual.subject, gold.object) &&
		matchesEntity(actual.object, gold.subject)
	);
}

function sameProvenance(actual: EvaluatedGraphAssertion, gold: GoldAssertion): boolean {
	return actual.chunkId === gold.chunkId && actual.documentId === gold.documentId;
}

function matchesEntity(actual: string, gold: GoldEntity): boolean {
	const candidate = normalize(actual);
	return [gold.canonical, ...(gold.aliases ?? [])].some((name) => normalize(name) === candidate);
}

function normalize(value: string): string {
	return value
		.toLocaleLowerCase()
		.replace(/[^\p{L}\p{N}]+/gu, ' ')
		.trim();
}

function ratio(numerator: number, denominator: number): number {
	return denominator === 0 ? 0 : numerator / denominator;
}
