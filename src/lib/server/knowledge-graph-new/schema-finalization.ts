import type { CorpusSchema, SchemaCategory } from './extraction';
import { isSemanticRelationCategory } from './assertion-quality';

export interface FinalizeCorpusSchemaOptions {
	universalEntityTypes: SchemaCategory[];
	proposedEntityTypes: SchemaCategory[];
	proposedRelationTypes: SchemaCategory[];
	sampledChunkIds: string[];
	version: string;
	maxEntityTypes: number;
	maxRelationTypes: number;
}

const BLOCKED_ENTITY_TYPES = new Set(['literal', 'other', 'string', 'text', 'unknown', 'value']);
const BLOCKED_RELATIONS = new Set(['co_occurs_with', 'cooccurs_with', 'related_to']);

/**
 * Finalize model-proposed schema data with deterministic guarantees.
 *
 * Relation endpoint types are prioritized over unused proposed types so every
 * retained relation is closed over the final entity vocabulary. A relation
 * endpoint omitted by the model's entity list is promoted into the vocabulary
 * instead of leaving an internally inconsistent schema.
 */
export function finalizeCorpusSchema(options: FinalizeCorpusSchemaOptions): CorpusSchema {
	validateLimits(options);

	const universal = uniqueEntityTypes(options.universalEntityTypes);
	if (universal.length > options.maxEntityTypes) {
		throw new Error('The entity-type limit is smaller than the universal vocabulary.');
	}

	const proposals = uniqueEntityTypes(options.proposedEntityTypes).filter(
		(type) => !universal.some((universalType) => universalType.name === type.name)
	);
	const proposedByName = new Map(proposals.map((type) => [type.name, type]));
	const relations = mergeRelations(options.proposedRelationTypes)
		.map(addRoleHolderCompatibility)
		.slice(0, options.maxRelationTypes);
	const entityTypes = [...universal];
	const entityNames = new Set(entityTypes.map((type) => type.name));

	// Relation endpoints carry more retrieval value than types that are never
	// used by a relation, so reserve the limited edge-device vocabulary for them.
	for (const relation of relations) {
		for (const endpointType of [
			...(relation.subjectTypes ?? []),
			...(relation.objectTypes ?? [])
		]) {
			if (entityNames.has(endpointType) || entityTypes.length >= options.maxEntityTypes) continue;
			const proposal = proposedByName.get(endpointType);
			entityTypes.push(
				proposal ?? {
					name: endpointType,
					description: `A corpus-specific entity used by the ${relation.name} relation`,
					source: 'relation-endpoint'
				}
			);
			entityNames.add(endpointType);
		}
	}

	for (const proposal of proposals) {
		if (entityTypes.length >= options.maxEntityTypes) break;
		if (entityNames.has(proposal.name)) continue;
		entityTypes.push(proposal);
		entityNames.add(proposal.name);
	}

	const relationTypes = relations.filter(
		(relation) =>
			(relation.subjectTypes ?? []).every((type) => entityNames.has(type)) &&
			(relation.objectTypes ?? []).every((type) => entityNames.has(type))
	);

	if (!relationTypes.length) {
		throw new Error('Schema discovery returned no closed, usable relation types.');
	}

	return {
		entityTypes,
		relationTypes,
		sampledChunkIds: [...new Set(options.sampledChunkIds)],
		version: options.version
	};
}

export function relationAcceptsTypes(
	relation: SchemaCategory,
	subjectType: string,
	objectType: string
): boolean {
	const normalizedSubject = normalizeCategoryName(subjectType);
	const normalizedObject = normalizeCategoryName(objectType);
	if (BLOCKED_ENTITY_TYPES.has(normalizedSubject) || BLOCKED_ENTITY_TYPES.has(normalizedObject)) {
		return false;
	}

	return (
		(relation.subjectTypes ?? []).includes(normalizedSubject) &&
		(relation.objectTypes ?? []).includes(normalizedObject)
	);
}

function uniqueEntityTypes(types: SchemaCategory[]): SchemaCategory[] {
	const byName = new Map<string, SchemaCategory>();

	for (const type of types) {
		const name = normalizeCategoryName(type.name);
		if (!name || BLOCKED_ENTITY_TYPES.has(name) || byName.has(name)) continue;
		const description = type.description.trim();
		if (!description) continue;
		byName.set(name, {
			name,
			description,
			source: type.source.trim() || 'schema-discovery'
		});
	}

	return [...byName.values()];
}

function mergeRelations(types: SchemaCategory[]): SchemaCategory[] {
	const byName = new Map<string, SchemaCategory>();

	for (const type of types) {
		const name = normalizeCategoryName(type.name);
		if (!name || BLOCKED_RELATIONS.has(name)) continue;
		const description = type.description.trim();
		const subjectTypes = endpointTypes(type.subjectTypes);
		const objectTypes = endpointTypes(type.objectTypes);
		if (!description || !subjectTypes.length || !objectTypes.length) continue;
		if (!isSemanticRelationCategory({ ...type, name, subjectTypes, objectTypes })) continue;

		const existing = byName.get(name);
		if (!existing) {
			byName.set(name, {
				name,
				description,
				source: type.source.trim() || 'schema-discovery',
				subjectTypes,
				objectTypes
			});
			continue;
		}
		if (existing.source === 'llm-consolidated') continue;

		existing.subjectTypes = [...new Set([...(existing.subjectTypes ?? []), ...subjectTypes])];
		existing.objectTypes = [...new Set([...(existing.objectTypes ?? []), ...objectTypes])];
	}

	return mergeEquivalentRelations([...byName.values()]);
}

function mergeEquivalentRelations(relations: SchemaCategory[]): SchemaCategory[] {
	const merged: SchemaCategory[] = [];

	for (const relation of relations) {
		const existing = merged.find((candidate) => equivalentRelations(candidate, relation));
		if (!existing) {
			merged.push(relation);
			continue;
		}

		existing.subjectTypes = [
			...new Set([...(existing.subjectTypes ?? []), ...(relation.subjectTypes ?? [])])
		];
		existing.objectTypes = [
			...new Set([...(existing.objectTypes ?? []), ...(relation.objectTypes ?? [])])
		];
	}

	return merged;
}

function equivalentRelations(left: SchemaCategory, right: SchemaCategory): boolean {
	if (!endpointTypesOverlap(left.subjectTypes, right.subjectTypes)) return false;
	if (!endpointTypesOverlap(left.objectTypes, right.objectTypes)) return false;

	const leftName = normalizeCategoryName(left.name);
	const rightName = normalizeCategoryName(right.name);
	if (stripLeadingAuxiliary(leftName) === stripLeadingAuxiliary(rightName)) return true;
	if (stripTrailingPreposition(leftName) === rightName) return true;
	if (stripTrailingPreposition(rightName) === leftName) return true;
	if (stripDocumentQualifier(leftName) === rightName) return true;
	if (stripDocumentQualifier(rightName) === leftName) return true;

	return descriptionSimilarity(left.description, right.description) >= 0.7;
}

function stripLeadingAuxiliary(value: string): string {
	return value.replace(/^(?:is|are|was|were)_/, '');
}

function addRoleHolderCompatibility(relation: SchemaCategory): SchemaCategory {
	return {
		...relation,
		subjectTypes: addPersonGroupForRole(relation.subjectTypes),
		objectTypes: addPersonGroupForRole(relation.objectTypes)
	};
}

function addPersonGroupForRole(types: string[] | undefined): string[] {
	const expanded = [...(types ?? [])];
	if (expanded.includes('role') && !expanded.includes('person_group')) {
		expanded.push('person_group');
	}
	return expanded;
}

function endpointTypesOverlap(left: string[] | undefined, right: string[] | undefined): boolean {
	if (!left?.length || !right?.length) return false;
	const normalized = new Set(left.map(normalizeCategoryName));
	return right.some((type) => normalized.has(normalizeCategoryName(type)));
}

function stripTrailingPreposition(value: string): string {
	return value.replace(/_(?:in|by|of|to)$/, '');
}

function stripDocumentQualifier(value: string): string {
	return value.replace(/_(?:policy|document|manual|regulation|directive|standard)$/, '');
}

function descriptionSimilarity(left: string, right: string): number {
	const leftWords = descriptionWords(left);
	const rightWords = descriptionWords(right);
	if (!leftWords.size || !rightWords.size) return 0;
	const intersection = [...leftWords].filter((word) => rightWords.has(word)).length;
	const union = new Set([...leftWords, ...rightWords]).size;
	return intersection / union;
}

function descriptionWords(value: string): Set<string> {
	return new Set(
		(value.toLowerCase().match(/[a-z][a-z0-9]{2,}/g) ?? []).filter(
			(word) => !['the', 'and', 'for', 'that', 'with', 'within'].includes(word)
		)
	);
}

function endpointTypes(types: string[] | undefined): string[] {
	if (!types) return [];
	return [
		...new Set(
			types.map(normalizeCategoryName).filter((type) => type && !BLOCKED_ENTITY_TYPES.has(type))
		)
	];
}

function normalizeCategoryName(value: string): string {
	return value
		.trim()
		.replace(/([a-z])([A-Z])/g, '$1_$2')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '_')
		.replace(/^_+|_+$/g, '');
}

function validateLimits(options: FinalizeCorpusSchemaOptions): void {
	if (!Number.isInteger(options.maxEntityTypes) || options.maxEntityTypes <= 0) {
		throw new Error('The entity-type limit must be a positive integer.');
	}
	if (!Number.isInteger(options.maxRelationTypes) || options.maxRelationTypes <= 0) {
		throw new Error('The relation-type limit must be a positive integer.');
	}
}
