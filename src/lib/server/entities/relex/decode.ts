import type { RelexWord } from './processor';
import type { RelexForward } from './runtime';

export interface RelexEntity {
	start: number;
	end: number;
	text: string;
	label: string;
	score: number;
}

export interface RelexRelation {
	head: RelexEntity;
	relation: string;
	tail: RelexEntity;
	score: number;
}

interface WordSpan {
	start: number;
	end: number;
	label: string;
	score: number;
}

function sigmoid(value: number): number {
	return 1 / (1 + Math.exp(-value));
}

function overlaps(left: WordSpan, right: WordSpan): boolean {
	if (left.start === right.start && left.end === right.end) return true;
	return !(left.start > right.end || right.start > left.end);
}

function findCandidates(
	forward: RelexForward,
	labels: readonly string[],
	threshold: number
): WordSpan[] {
	const candidates: WordSpan[] = [];
	for (let start = 0; start < forward.words; start += 1) {
		for (let width = 0; width < forward.maxWidth; width += 1) {
			if (start + width + 1 > forward.words) continue;
			const offset = (start * forward.maxWidth + width) * forward.entityClasses;
			for (let label = 0; label < forward.entityClasses; label += 1) {
				const score = sigmoid(forward.scores[offset + label]);
				if (score > threshold)
					candidates.push({ start, end: start + width, label: labels[label], score });
			}
		}
	}
	return candidates;
}

function greedyFlat(candidates: WordSpan[]): WordSpan[] {
	const selected: WordSpan[] = [];
	for (const candidate of [...candidates].sort((left, right) => right.score - left.score)) {
		if (!selected.some((span) => overlaps(candidate, span))) selected.push(candidate);
	}
	return selected.sort((left, right) => left.start - right.start);
}

function toEntity(span: WordSpan, words: readonly RelexWord[], text: string): RelexEntity {
	const start = words[span.start].start;
	const end = words[span.end].end;
	return { start, end, text: text.slice(start, end), label: span.label, score: span.score };
}

function modelToDecodedIndex(
	forward: RelexForward,
	spans: readonly WordSpan[]
): Map<number, number> {
	const byBoundary = new Map<string, number>();
	spans.forEach((span, index) => {
		const key = `${span.start}:${span.end}`;
		if (!byBoundary.has(key)) byBoundary.set(key, index);
	});
	const mapping = new Map<number, number>();
	forward.entitySpans.forEach(([start, end], entity) => {
		const decoded = byBoundary.get(`${start}:${end}`);
		if (decoded !== undefined) mapping.set(entity, decoded);
	});
	return mapping;
}

function decodeRelations(
	forward: RelexForward,
	spans: readonly WordSpan[],
	entities: readonly RelexEntity[],
	relationLabels: readonly string[],
	threshold: number
): RelexRelation[] {
	const mapping = modelToDecodedIndex(forward, spans);
	const relations: RelexRelation[] = [];
	forward.pairs.forEach(([modelHead, modelTail], pair) => {
		const head = mapping.get(modelHead);
		const tail = mapping.get(modelTail);
		if (head === undefined || tail === undefined) return;
		for (let label = 0; label < forward.relationClasses; label += 1) {
			const score = sigmoid(forward.relationLogits[pair * forward.relationClasses + label]);
			if (score <= threshold) continue;
			relations.push({
				head: entities[head],
				relation: relationLabels[label],
				tail: entities[tail],
				score
			});
		}
	});
	return relations;
}

export function decodeRelex(
	forward: RelexForward,
	words: readonly RelexWord[],
	text: string,
	labels: { entity: readonly string[]; relation: readonly string[] },
	thresholds: { entity: number; relation: number }
): { entities: RelexEntity[]; relations: RelexRelation[] } {
	const spans = greedyFlat(findCandidates(forward, labels.entity, thresholds.entity));
	const entities = spans.map((span) => toEntity(span, words, text));
	return {
		entities,
		relations: decodeRelations(forward, spans, entities, labels.relation, thresholds.relation)
	};
}
