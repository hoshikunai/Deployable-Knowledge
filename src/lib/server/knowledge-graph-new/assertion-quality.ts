import type { CorpusSchema, ExtractedAssertion, SchemaCategory } from './extraction';

const DOCUMENT_TYPE_PATTERN = /(?:^|_)(?:document|publication|manual|directive)(?:_|$)/;
const PROVENANCE_RELATION_PATTERN =
	/(?:^|_)(?:document(?:s|ed)?|detail(?:s|ed)?|describ(?:e|es|ed)|contain(?:s|ed)?|includ(?:e|es|ed)|list(?:s|ed)?|mention(?:s|ed)?|appear(?:s|ed)?|references?|cites?|recorded_in|issued_(?:by|on))(?:_|$)/;
const PROVENANCE_DESCRIPTION_PATTERN =
	/(?:^|_)(?:detail(?:s|ed)?|describ(?:e|es|ed)|contain(?:s|ed)?|includ(?:e|es|ed)|list(?:s|ed)?|mention(?:s|ed)?|appear(?:s|ed)?|references?|cites?|recorded_in|issued_(?:by|on))(?:_|$)/;
const VAGUE_RELATION_PATTERN = /(?:^|_)(?:related_to|associated_with)(?:_|$)/;
const PASSIVE_USE_RELATION_PATTERN = /(?:^|_)is_used_(?:for|by)(?:_|$)/;
const ACTOR_ENTITY_TYPES = new Set(['person', 'role', 'person_group']);
const CLAUSE_VERB_PATTERN =
	/\b(?:is|are|was|were|be|been|being|must|shall|should|will|would|can|could|may|might|has|have|had|designated|described|required|authorized|prohibited|approved|performed|conducted|assigned)\b/i;
const CITATION_OPENING_PATTERN = /^\s*(?:see|refer to|consult)\b/i;
const CITATION_CLOSING_PATTERN = /\bfor (?:more|additional) (?:information|guidance)\b/i;
const CITATION_RELATION_PATTERN = /(?:^|_)(?:references?|cites?)(?:_|$)/;
const DOCUMENT_LOCATOR_PATTERN =
	/^(?:paragraph|para\.?|section|chapter|appendix|figure|table)\s+(?:[a-z]|\d)[a-z0-9().-]*$/i;
const TEMPLATE_PATTERNS = [
	/\bopening sentence\b/gi,
	/\bnarrative description\b/gi,
	/\bclosing sentence\b/gi,
	/_{3,}/g,
	/\((?:date|location|name|rank|duty assignment|at or near)\)/gi
];
const PROHIBITED_MODALITY_PATTERN =
	/\b(?:must|shall|may)\s+not\b|\b(?:prohibited|forbidden)\b|\bnot\s+(?:permitted|authorized|allowed)\b/i;
const REQUIRED_MODALITY_PATTERN =
	/\b(?:must|shall|will)\b|\b(?:mandatory|required)\b|\b(?:has|have|had)\s+to\b|\bneed(?:s|ed)?\s+to\b/i;
const RECOMMENDED_MODALITY_PATTERN = /\bshould\b|\bought\s+to\b|\brecommend(?:ed|s)?\b/i;
const PERMITTED_MODALITY_PATTERN =
	/\bmay\b|\b(?:permitted|allowed|optional)\b|\bauthoriz(?:e|ed|es)\b/i;
const HABITUAL_MODALITY_PATTERN =
	/\b(?:usually|generally|routinely|regularly|customarily|typically)\b/i;
const EPISTEMIC_MODALITY_PATTERN =
	/\b(?:may|might|could|possibly|possible|probably|likely|unlikely)\b/i;
const ABILITY_OR_FUTURE_MODALITY_PATTERN = /\b(?:can|will)\b/i;
const CONDITION_PATTERN =
	/\b(?:if|unless|provided\s+that|as\s+long\s+as|only\s+when|when\s+(?:necessary|required)|as\s+needed|except\s+when)\b/i;

/**
 * Keep document structure in provenance metadata instead of semantic graph
 * edges. This classification is based on relation semantics and endpoint
 * categories, not corpus-specific vocabulary.
 */
export function isSemanticRelationCategory(relation: SchemaCategory): boolean {
	const relationName = normalizeCategoryName(relation.name);
	if (VAGUE_RELATION_PATTERN.test(relationName)) return false;
	if (
		PASSIVE_USE_RELATION_PATTERN.test(relationName) &&
		(relation.subjectTypes ?? []).some((type) =>
			ACTOR_ENTITY_TYPES.has(normalizeCategoryName(type))
		)
	) {
		return false;
	}

	const endpointTypes = [...(relation.subjectTypes ?? []), ...(relation.objectTypes ?? [])];
	const hasDocumentEndpoint = endpointTypes.some((type) =>
		DOCUMENT_TYPE_PATTERN.test(normalizeCategoryName(type))
	);

	const description = normalizeCategoryName(relation.description);
	return !(
		hasDocumentEndpoint &&
		(PROVENANCE_RELATION_PATTERN.test(relationName) ||
			PROVENANCE_DESCRIPTION_PATTERN.test(description))
	);
}

export function isSemanticAssertionCandidate(
	text: string,
	assertion: ExtractedAssertion,
	schema: CorpusSchema
): boolean {
	const relation = schema.relationTypes.find(
		(type) => normalizeCategoryName(type.name) === normalizeCategoryName(assertion.rawPredicate)
	);
	if (!relation || !isSemanticRelationCategory(relation)) return false;
	if (looksLikeTemplate(text)) return false;
	if (looksLikeClause(assertion.subject) || looksLikeClause(assertion.object)) return false;
	if (isGroupTypedAsPerson(assertion.subject, assertion.subjectType)) return false;
	if (isGroupTypedAsPerson(assertion.object, assertion.objectType)) return false;
	if (isDocumentLocator(assertion.subject, assertion.subjectType)) return false;
	if (isDocumentLocator(assertion.object, assertion.objectType)) return false;
	if (
		!hasConsistentModality(
			assertion.evidence,
			assertion.modality,
			assertion.modalityCue,
			assertion.status,
			assertion.condition
		)
	) {
		return false;
	}

	const predicate = normalizeCategoryName(assertion.rawPredicate);
	if (
		CITATION_OPENING_PATTERN.test(assertion.evidence) &&
		CITATION_CLOSING_PATTERN.test(assertion.evidence) &&
		!CITATION_RELATION_PATTERN.test(predicate)
	) {
		return false;
	}

	return true;
}

export function hasConsistentModality(
	evidence: string,
	modality: ExtractedAssertion['modality'],
	modalityCue: string | null,
	status: ExtractedAssertion['status'] = 'asserted',
	condition: string | null = null
): boolean {
	if (!hasConsistentCondition(evidence, condition)) return false;
	if (modalityCue !== null && !evidence.includes(modalityCue)) return false;

	if (modality === 'prohibited') {
		return (
			status === 'asserted' && modalityCue !== null && PROHIBITED_MODALITY_PATTERN.test(modalityCue)
		);
	}
	if (modality === 'required') {
		return (
			modalityCue !== null &&
			!PROHIBITED_MODALITY_PATTERN.test(modalityCue) &&
			REQUIRED_MODALITY_PATTERN.test(modalityCue)
		);
	}
	if (modality === 'recommended') {
		return modalityCue !== null && RECOMMENDED_MODALITY_PATTERN.test(modalityCue);
	}
	if (modality === 'permitted') {
		return (
			modalityCue !== null &&
			!PROHIBITED_MODALITY_PATTERN.test(modalityCue) &&
			PERMITTED_MODALITY_PATTERN.test(modalityCue)
		);
	}
	if (modality === 'habitual') {
		return modalityCue === null
			? !hasUnrepresentedExplicitModality(evidence)
			: HABITUAL_MODALITY_PATTERN.test(modalityCue);
	}
	if (modalityCue !== null) {
		if (EPISTEMIC_MODALITY_PATTERN.test(modalityCue)) return status === 'uncertain';
		return ABILITY_OR_FUTURE_MODALITY_PATTERN.test(modalityCue);
	}

	return !hasUnrepresentedExplicitModality(evidence);
}

export function hasConsistentCondition(evidence: string, condition: string | null): boolean {
	if (condition !== null && !evidence.includes(condition)) return false;
	if (!CONDITION_PATTERN.test(evidence)) return true;
	return condition !== null && CONDITION_PATTERN.test(condition);
}

function hasUnrepresentedExplicitModality(evidence: string): boolean {
	return (
		PROHIBITED_MODALITY_PATTERN.test(evidence) ||
		REQUIRED_MODALITY_PATTERN.test(evidence) ||
		RECOMMENDED_MODALITY_PATTERN.test(evidence) ||
		PERMITTED_MODALITY_PATTERN.test(evidence) ||
		HABITUAL_MODALITY_PATTERN.test(evidence) ||
		EPISTEMIC_MODALITY_PATTERN.test(evidence) ||
		ABILITY_OR_FUTURE_MODALITY_PATTERN.test(evidence)
	);
}

export function isDocumentLocator(value: string, type: string): boolean {
	return (
		DOCUMENT_TYPE_PATTERN.test(normalizeCategoryName(type)) &&
		DOCUMENT_LOCATOR_PATTERN.test(value.trim())
	);
}

export function looksLikeTemplate(text: string): boolean {
	let markers = 0;

	for (const pattern of TEMPLATE_PATTERNS) {
		pattern.lastIndex = 0;
		markers += [...text.matchAll(pattern)].length;
		if (markers >= 2) return true;
	}

	return false;
}

export function looksLikeClause(value: string): boolean {
	const words = value.match(/[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu) ?? [];
	return words.length >= 6 && CLAUSE_VERB_PATTERN.test(value);
}

export function isGroupTypedAsPerson(value: string, type: string): boolean {
	if (normalizeCategoryName(type) !== 'person') return false;
	const clean = value.trim();
	if (/^(?:all|any|each|every|multiple|several)\b/i.test(clean)) return true;
	if (/^[A-Z]{2,}s$/.test(clean)) return true;

	const words = clean.match(/[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu) ?? [];
	const lastWord = words.at(-1) ?? '';
	return words.length >= 2 && /^[a-z].*s$/u.test(lastWord);
}

function normalizeCategoryName(value: string): string {
	return value
		.trim()
		.replace(/([a-z])([A-Z])/g, '$1_$2')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '_')
		.replace(/^_+|_+$/g, '');
}
