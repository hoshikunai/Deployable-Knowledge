import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { Parser, Store } from 'n3';
import { getProvider } from '$lib/server/providers/registry';
import { estimateTokens } from '$lib/utils/tokens';
import { isSemanticAssertionCandidate } from './assertion-quality';
import { finalizeCorpusSchema, relationAcceptsTypes } from './schema-finalization';
import {
	evaluateSchemaQuality,
	schemaQualityFailureMessage,
	type SchemaQualityReport
} from './schema-quality';
import { buildSchemaDiscoveryBatches, buildSchemaSample } from './schema-sampling';
import type { ProviderChatOptions } from '$lib/server/providers/provider';

export type GraphChunk = {
	chunkId: string;
	documentId: string;
	content: string;
};

export type SchemaCategory = {
	name: string;
	description: string;
	source: string;
	subjectTypes?: string[];
	objectTypes?: string[];
};

export type CorpusSchema = {
	entityTypes: SchemaCategory[];
	relationTypes: SchemaCategory[];
	sampledChunkIds: string[];
	version: string;
	schemaProvenance?: {
		discoveryBatches: number;
		entityConsolidation: 'complete' | 'failed';
		relationConsolidation: 'complete' | 'chunked' | 'failed';
		fallbackUsed: boolean;
		warnings: string[];
		qualityGate: SchemaQualityReport;
	};
};

export type Extractor = 'llm' | 'gliner';
export type AssertionModality =
	| 'observed'
	| 'habitual'
	| 'required'
	| 'recommended'
	| 'permitted'
	| 'prohibited';

export type ExtractedAssertion = {
	subject: string;
	subjectType: string;
	rawPredicate: string;
	object: string;
	objectType: string;
	evidence: string;
	startDate: string | null;
	endDate: string | null;
	status: 'asserted' | 'negated' | 'uncertain';
	modality: AssertionModality;
	modalityCue: string | null;
	condition: string | null;
	extractors: Extractor[];
	verified: boolean;
	score: number | null;
	offsets: [number, number, number, number] | null;
};

export type ExtractedEntity = {
	text: string;
	type: string;
	start: number;
	end: number;
	score: number | null;
};

export type ExtractionResult = {
	assertions: ExtractedAssertion[];
	entities?: ExtractedEntity[];
};

export type ExtractionSettings = {
	providerId: string;
	modelId: string;
	providerOptions?: ProviderChatOptions;
	useGliner?: boolean;
};

type ScoredTerm = SchemaCategory & { relation: boolean; score: number };

const VERSION = 'kg-v20';
// 16,384 for 12 GiB RAM
const DEFAULT_CONTEXT_WINDOW = 16_384;
const DEFAULT_SCHEMA_SAMPLE_CHUNKS = 30;
const DEFAULT_SCHEMA_SAMPLE_CHARACTERS = 24_000;
const DEFAULT_SCHEMA_DISCOVERY_BATCHES = 3;
const DEFAULT_SCHEMA_BATCH_CHARACTERS = 8_000;
const DEFAULT_MAX_ENTITY_TYPES = 24;
const DEFAULT_MAX_RELATION_TYPES = 24;
const RELATION_CONSOLIDATION_GROUP_SIZE = 8;
const STRUCTURED_OUTPUT_CONTEXT_RESERVE = 512;
const DEFAULT_SCHEMA_QUALITY_THRESHOLD = 0.8;

const ONTOLOGIES = [
	'https://schema.org/version/latest/schemaorg-current-https.ttl',
	'https://www.w3.org/ns/prov.ttl'
];
const UNIVERSAL: SchemaCategory[] = [
	['person', 'A named individual'],
	['organization', 'A named organization or institution'],
	['location', 'A named geographic or physical place'],
	['event', 'A named historical, scientific, or operational event'],
	['document', 'A named publication, law, treaty, or standard'],
	['date', 'A date or named time period'],
	['system', 'A named technical, organizational, or conceptual system'],
	['component', 'A named part of a larger system'],
	['process', 'A named procedure, operation, or method'],
	['object', 'A named physical or conceptual object'],
	['role', 'A named or defined occupational, professional, or functional role'],
	['person_group', 'A named or explicitly defined group or class of people']
].map(([name, description]) => ({ name, description, source: 'universal' }));

const string = { type: 'string' };
const nullableString = { type: ['string', 'null'] };
const object = (properties: Record<string, unknown>) => ({
	type: 'object',
	additionalProperties: false,
	properties,
	required: Object.keys(properties)
});
const array = (items: unknown, maxItems?: number) => ({
	type: 'array',
	items,
	...(maxItems ? { maxItems } : {})
});

const booleanValue = { type: 'boolean' };
const modalityValue = {
	type: 'string',
	enum: ['observed', 'habitual', 'required', 'recommended', 'permitted', 'prohibited']
};

const VERIFICATION_OUTPUT = object({
	decisions: array(
		object({
			index: { type: 'integer' },
			verdict: {
				type: 'string',
				enum: ['accept', 'reverse', 'reject']
			},
			subjectTypeCorrect: booleanValue,
			objectTypeCorrect: booleanValue,
			modalityCorrect: booleanValue,
			conditionCorrect: booleanValue
		})
	)
});

function schemaOutput(maxEntityTypes: number, maxRelationTypes: number) {
	return object({
		entityTypes: array(object({ name: string, description: string }), maxEntityTypes),
		relationTypes: array(
			object({
				name: string,
				description: string,
				subjectTypes: array(string),
				objectTypes: array(string)
			}),
			maxRelationTypes
		)
	});
}

function entityTypesOutput(maxEntityTypes: number) {
	return object({
		entityTypes: array(object({ name: string, description: string }), maxEntityTypes)
	});
}

function relationTypesOutput(maxRelationTypes: number) {
	return object({
		relationTypes: array(
			object({
				name: string,
				description: string,
				subjectTypes: array(string),
				objectTypes: array(string)
			}),
			maxRelationTypes
		)
	});
}

export function buildExtractionOutput(schema: CorpusSchema) {
	if (!schema.relationTypes.length) {
		throw new Error('Extraction requires at least one canonical relation.');
	}

	const assertionBranches = schema.relationTypes.map((relation) =>
		object({
			subject: string,
			subjectType: { type: 'string', enum: relation.subjectTypes },
			rawPredicate: { type: 'string', const: relation.name },
			object: string,
			objectType: { type: 'string', enum: relation.objectTypes },
			evidence: string,
			startDate: nullableString,
			endDate: nullableString,
			status: { type: 'string', enum: ['asserted', 'negated', 'uncertain'] },
			modality: modalityValue,
			modalityCue: nullableString,
			condition: nullableString
		})
	);

	return object({ assertions: array({ oneOf: assertionBranches }, 30) });
}

function discoveryPrompt(sample: string, ontologyTerms: SchemaCategory[]): string {
	return `Propose an extraction schema for this diverse slice of an unfamiliar corpus.

Existing universal entity types:
${JSON.stringify(UNIVERSAL.map(({ name, description }) => ({ name, description })))}

Potential established ontology terms, included only as optional hints:
${JSON.stringify(
	ontologyTerms.map(({ name, description, subjectTypes, objectTypes }) => ({
		name,
		description,
		subjectTypes,
		objectTypes
	}))
)}

Return reusable corpus-specific entity types that add meaning beyond the
universal types. Never return entity names, acronyms, headings, document titles,
or one-off phrases as types. Return precise directed relations that are explicitly
supported by the sample. Relation names must be stable snake_case predicates,
not sentence fragments, and must read grammatically as “subject predicate
object.” For example, an actor that uses a method needs an active predicate such
as uses_method, not a passive is_used_for predicate. Every relation must declare
nonempty subjectTypes and objectTypes using universal types or corpus-specific
types in this response.
Use person only for named individuals, role for occupational or functional roles,
and person_group for groups or classes of people. Never create related_to,
co-occurrence, page-layout, appendix, figure, table, is_detailed_in, or
is_documented_by relations. Document location belongs in provenance metadata.
Do not treat a cited document title as a process. Prefer omitting a relationship
over forcing it into an inaccurate predicate.

Corpus slice:
${sample}`;
}

function entityConsolidationPrompt(entityTypes: SchemaCategory[]): string {
	return `Consolidate entity-type candidates from independently sampled corpus slices.

Universal entity types, which must not be returned:
${JSON.stringify(UNIVERSAL.map(({ name, description }) => ({ name, description })))}

Additional candidates:
${JSON.stringify(entityTypes.map(({ name, description, source }) => ({ name, description, source })))}

Merge synonyms under stable snake_case names. Preserve reusable distinctions
that materially affect retrieval even when they occur in one slice. Exclude
proper nouns, acronyms, headings, layout elements, and overly broad duplicates
of universal types. Return only additional reusable entity types.`;
}

function relationConsolidationPrompt(
	relationTypes: SchemaCategory[],
	allowedEntityTypes: string[]
): string {
	return `Consolidate directed relation candidates from a heterogeneous corpus.

Allowed endpoint entity types:
${JSON.stringify(allowedEntityTypes)}

Relation candidates:
${JSON.stringify(
	relationTypes.map(({ name, description, subjectTypes, objectTypes, source }) => ({
		name,
		description,
		subjectTypes,
		objectTypes,
		source
	}))
)}

Merge synonyms under stable snake_case predicate names. Preserve reusable
distinctions that materially affect retrieval, including requirements,
permissions, composition, compatibility, and responsibilities. Every relation
must read grammatically as “subject predicate object” and declare nonempty
subjectTypes and objectTypes using only allowed endpoint types. Never assign an
actor subject to a passive is_used_for predicate. Exclude related_to,
co-occurrence, document-layout, citation-only, is_detailed_in, and
is_documented_by relations. Omit a relation rather than changing its modality or
forcing it into a generic predicate.`;
}

async function consolidateSchemaCandidates(
	settings: ExtractionSettings,
	entityTypes: SchemaCategory[],
	relationTypes: SchemaCategory[],
	maxEntityTypes: number,
	maxRelationTypes: number
): Promise<{
	entityTypes: SchemaCategory[];
	relationTypes: SchemaCategory[];
	entityStatus: 'complete' | 'failed';
	relationStatus: 'complete' | 'chunked' | 'failed';
	warnings: string[];
}> {
	const warnings: string[] = [];
	const additionalEntityLimit = Math.max(1, maxEntityTypes - UNIVERSAL.length);
	let consolidatedEntityTypes: SchemaCategory[] = [];
	let entityStatus: 'complete' | 'failed' = 'complete';

	try {
		const result = await askJson<{ entityTypes: unknown }>(
			settings,
			entityConsolidationPrompt(entityTypes),
			entityTypesOutput(additionalEntityLimit),
			schemaTokenBudget(additionalEntityLimit, 0),
			{ thinking: true }
		);
		consolidatedEntityTypes = readCategories(result.entityTypes).map((type) => ({
			...type,
			source: 'llm-consolidated'
		}));
	} catch (error) {
		entityStatus = 'failed';
		warnings.push(`Entity consolidation failed: ${message(error)}`);
	}

	const allowedEntityTypes = [
		...new Set([
			...UNIVERSAL.map((type) => type.name),
			...entityTypes.map((type) => category(type.name)),
			...consolidatedEntityTypes.map((type) => type.name)
		])
	].filter(Boolean);
	let consolidatedRelationTypes: SchemaCategory[] = [];
	let relationStatus: 'complete' | 'chunked' | 'failed' = 'complete';

	try {
		const result = await askJson<{ relationTypes: unknown }>(
			settings,
			relationConsolidationPrompt(relationTypes, allowedEntityTypes),
			relationTypesOutput(maxRelationTypes),
			schemaTokenBudget(0, maxRelationTypes),
			{ thinking: true }
		);
		consolidatedRelationTypes = readCategories(result.relationTypes, true).map((type) => ({
			...type,
			source: 'llm-consolidated'
		}));
		if (!consolidatedRelationTypes.length) {
			throw new Error('Relation consolidation returned no usable relations.');
		}
	} catch (error) {
		warnings.push(`Full relation consolidation failed: ${message(error)}`);
		relationStatus = 'chunked';
		const groups = relationConsolidationGroups(relationTypes);

		for (const [index, group] of groups.entries()) {
			try {
				const result = await askJson<{ relationTypes: unknown }>(
					settings,
					relationConsolidationPrompt(group, allowedEntityTypes),
					relationTypesOutput(Math.min(maxRelationTypes, group.length)),
					schemaTokenBudget(0, group.length),
					{ thinking: true }
				);
				consolidatedRelationTypes.push(
					...readCategories(result.relationTypes, true).map((type) => ({
						...type,
						source: `llm-consolidated-group-${index + 1}`
					}))
				);
			} catch (groupError) {
				warnings.push(`Relation consolidation group ${index + 1} failed: ${message(groupError)}`);
			}
		}

		if (!consolidatedRelationTypes.length) relationStatus = 'failed';
	}

	return {
		entityTypes: consolidatedEntityTypes,
		relationTypes: consolidatedRelationTypes,
		entityStatus,
		relationStatus,
		warnings
	};
}

function relationConsolidationGroups(relationTypes: SchemaCategory[]): SchemaCategory[][] {
	const ordered = [...relationTypes].sort((left, right) => {
		const leftEndpoints = `${(left.subjectTypes ?? []).join(',')}\u0000${(left.objectTypes ?? []).join(',')}`;
		const rightEndpoints = `${(right.subjectTypes ?? []).join(',')}\u0000${(right.objectTypes ?? []).join(',')}`;
		return leftEndpoints.localeCompare(rightEndpoints) || left.name.localeCompare(right.name);
	});
	const groups: SchemaCategory[][] = [];
	for (let index = 0; index < ordered.length; index += RELATION_CONSOLIDATION_GROUP_SIZE) {
		groups.push(ordered.slice(index, index + RELATION_CONSOLIDATION_GROUP_SIZE));
	}
	return groups;
}

function schemaTokenBudget(maxEntityTypes: number, maxRelationTypes: number): number {
	return Math.min(7_000, 1_000 + maxEntityTypes * 80 + maxRelationTypes * 180);
}

function recurrentCategories(types: SchemaCategory[]): SchemaCategory[] {
	const sources = new Map<string, Set<string>>();
	for (const type of types) {
		const name = category(type.name);
		if (!name) continue;
		const supportingSources = sources.get(name) ?? new Set<string>();
		supportingSources.add(type.source);
		sources.set(name, supportingSources);
	}
	return unique(types.filter((type) => (sources.get(category(type.name))?.size ?? 0) >= 2));
}

function promptEntityTypes(schema: CorpusSchema) {
	return schema.entityTypes.map(({ name, description }) => ({ name, description }));
}

export function extractionVersion(): Record<string, unknown> {
	return {
		version: VERSION,
		contextWindow: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_CONTEXT_WINDOW,
			DEFAULT_CONTEXT_WINDOW
		),
		schemaSampleChunks: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_SAMPLE_CHUNKS,
			DEFAULT_SCHEMA_SAMPLE_CHUNKS
		),
		schemaSampleCharacters: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_SAMPLE_CHARACTERS,
			DEFAULT_SCHEMA_SAMPLE_CHARACTERS
		),
		schemaDiscoveryBatches: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_DISCOVERY_BATCHES,
			DEFAULT_SCHEMA_DISCOVERY_BATCHES
		),
		schemaBatchCharacters: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_BATCH_CHARACTERS,
			DEFAULT_SCHEMA_BATCH_CHARACTERS
		),
		maxEntityTypes: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_MAX_ENTITY_TYPES,
			DEFAULT_MAX_ENTITY_TYPES
		),
		maxRelationTypes: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_MAX_RELATION_TYPES,
			DEFAULT_MAX_RELATION_TYPES
		),
		structuredOutput: 'json-schema-v2-nonthinking',
		schemaConsolidation: 'staged-v1',
		assertionModality: 'cue-and-condition-v2',
		schemaQualityGate: 'deterministic-v1',
		model: process.env.KNOWLEDGE_GRAPH_GLINER_MODEL ?? 'knowledgator/gliner-relex-large-v0.5',
		entity: process.env.KNOWLEDGE_GRAPH_GLINER_THRESHOLD ?? '0.4',
		adjacency: process.env.KNOWLEDGE_GRAPH_GLINER_ADJACENCY_THRESHOLD ?? '0.55',
		relation: process.env.KNOWLEDGE_GRAPH_GLINER_RELATION_THRESHOLD ?? '0.75',
		other: process.env.KNOWLEDGE_GRAPH_GLINER_OTHER === 'true'
	};
}

export async function discoverCorpusSchema(
	chunks: GraphChunk[],
	settings: ExtractionSettings
): Promise<CorpusSchema> {
	const maxEntityTypes = positiveInteger(
		process.env.KNOWLEDGE_GRAPH_MAX_ENTITY_TYPES,
		DEFAULT_MAX_ENTITY_TYPES
	);
	const maxRelationTypes = positiveInteger(
		process.env.KNOWLEDGE_GRAPH_MAX_RELATION_TYPES,
		DEFAULT_MAX_RELATION_TYPES
	);
	const schemaSample = buildSchemaSample(chunks, {
		maxChunks: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_SAMPLE_CHUNKS,
			DEFAULT_SCHEMA_SAMPLE_CHUNKS
		),
		maxCharacters: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_SAMPLE_CHARACTERS,
			DEFAULT_SCHEMA_SAMPLE_CHARACTERS
		)
	});

	const sampled = schemaSample.chunks;
	if (!schemaSample.text) throw new Error('No usable text is available for schema discovery.');

	const discoveryBatches = buildSchemaDiscoveryBatches(sampled, {
		maxBatches: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_DISCOVERY_BATCHES,
			DEFAULT_SCHEMA_DISCOVERY_BATCHES
		),
		maxCharactersPerBatch: positiveInteger(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_BATCH_CHARACTERS,
			DEFAULT_SCHEMA_BATCH_CHARACTERS
		)
	});
	const ontologyTerms = (await loadOntologyTerms(schemaSample.text)).slice(0, 16);
	const proposals: Array<{ entityTypes: SchemaCategory[]; relationTypes: SchemaCategory[] }> = [];

	for (let index = 0; index < discoveryBatches.length; index += 1) {
		try {
			const result = await askJson<{ entityTypes: unknown; relationTypes: unknown }>(
				settings,
				discoveryPrompt(discoveryBatches[index], ontologyTerms),
				schemaOutput(10, 20),
				schemaTokenBudget(10, 20),
				{ thinking: true }
			);
			proposals.push({
				entityTypes: readCategories(result.entityTypes).map((type) => ({
					...type,
					source: `llm-discovery-${index + 1}`
				})),
				relationTypes: readCategories(result.relationTypes, true).map((type) => ({
					...type,
					source: `llm-discovery-${index + 1}`
				}))
			});
		} catch (error) {
			console.warn(`Schema discovery batch ${index + 1} failed: ${message(error)}`);
		}
	}

	if (!proposals.length) {
		throw new Error('All adaptive schema discovery batches failed.');
	}

	const allEntityTypes = proposals.flatMap((proposal) => proposal.entityTypes);
	const allRelationTypes = proposals.flatMap((proposal) => proposal.relationTypes);
	const recurrentEntityTypes = recurrentCategories(allEntityTypes);
	const recurrentRelationTypes = recurrentCategories(allRelationTypes);
	const consolidation = await consolidateSchemaCandidates(
		settings,
		allEntityTypes,
		allRelationTypes,
		maxEntityTypes,
		maxRelationTypes
	);
	for (const warning of consolidation.warnings) console.warn(warning);
	if (consolidation.relationStatus === 'failed') {
		throw new Error(
			`Schema relation consolidation failed; extraction was not started. ${consolidation.warnings.join(' ')}`
		);
	}

	const schema = finalizeCorpusSchema({
		universalEntityTypes: UNIVERSAL,
		proposedEntityTypes: unique([
			...consolidation.entityTypes,
			...recurrentEntityTypes,
			...allEntityTypes
		]),
		proposedRelationTypes: [
			...consolidation.relationTypes,
			...recurrentRelationTypes,
			...allRelationTypes
		],
		sampledChunkIds: sampled.map((chunk) => chunk.chunkId),
		version: VERSION,
		maxEntityTypes,
		maxRelationTypes
	});
	const qualityGate = evaluateSchemaQuality(schema, {
		discoveryProposals: proposals,
		entityConsolidation: consolidation.entityStatus,
		relationConsolidation: consolidation.relationStatus,
		minimumScore: boundedNumber(
			process.env.KNOWLEDGE_GRAPH_SCHEMA_MIN_QUALITY_SCORE,
			DEFAULT_SCHEMA_QUALITY_THRESHOLD,
			0,
			1
		)
	});
	for (const issue of qualityGate.issues) {
		if (issue.severity === 'warning') console.warn(`Schema quality warning: ${issue.message}`);
	}
	if (qualityGate.status === 'failed') {
		throw new Error(schemaQualityFailureMessage(qualityGate));
	}
	const qualityWarnings = qualityGate.issues
		.filter((issue) => issue.severity === 'warning')
		.map((issue) => `${issue.code}: ${issue.message}`);

	return {
		...schema,
		schemaProvenance: {
			discoveryBatches: proposals.length,
			entityConsolidation: consolidation.entityStatus,
			relationConsolidation: consolidation.relationStatus,
			fallbackUsed:
				consolidation.entityStatus === 'failed' || consolidation.relationStatus !== 'complete',
			warnings: [...consolidation.warnings, ...qualityWarnings],
			qualityGate
		}
	};
}

export async function extractWithLlm(
	chunk: GraphChunk,
	schema: CorpusSchema,
	settings: ExtractionSettings,
	entityHints: ExtractedEntity[] = []
): Promise<ExtractionResult> {
	const allowedRelations = schema.relationTypes.map((relation) => ({
		name: relation.name,
		description: relation.description,
		subjectTypes: relation.subjectTypes,
		objectTypes: relation.objectTypes
	}));
	const hints = selectEntityHints(chunk.content, schema, entityHints);
	const result = await askJson<Record<string, unknown>>(
		settings,
		`Extract a small, evidence-grounded knowledge graph from this chunk.

Allowed entity types:
${JSON.stringify(promptEntityTypes(schema))}

Allowed canonical relations:
${JSON.stringify(allowedRelations)}

rawPredicate must exactly match one allowed relation name. Do not invent
predicates. If an explicitly stated relationship cannot be represented by an
allowed relation, omit it from accepted assertions.

Potential entity spans from a local NER model:
${JSON.stringify(hints.map(({ text, type }) => ({ text, type })))}

These spans are untrusted hints, not assertions. Use a hint only when its exact
text and type are correct in context. You may extract an entity that is absent
from the hints, and you must ignore false, incomplete, or irrelevant hints.

Extract only explicitly stated, meaningful relationships. Copy subject, object,
and one evidence substring exactly from the chunk; the evidence must contain
both endpoints. Preserve direction, dates, negation, uncertainty, and the scope
of modal language.

Represent modality independently from status:
- observed: the relationship is stated as a fact or actual occurrence;
- habitual: the relationship is a recurring practice or responsibility;
- required: the relationship is obligatory (must, shall, mandatory, required);
- recommended: the relationship is advised (should, ought, recommended);
- permitted: the relationship grants authorization or choice (permitted,
  authorized, allowed, optional, or "may" when it grants permission);
- prohibited: the relationship is forbidden (must not, shall not, may not,
  prohibited, forbidden, or not permitted).

Copy the exact words that express modality into modalityCue. For an unmarked
observation, or a habitual assertion without an explicit cue, set modalityCue
to null. Bare "may" can express possibility instead of permission; in that case
use observed modality with uncertain status and still copy "may" as the cue. "Can" usually
expresses ability, not permission. "Will" is required only when a prescriptive
document uses it to impose an obligation; otherwise it describes a future fact.

If the relationship applies only under an explicit condition, copy the smallest
exact condition phrase (for example, "if approved", "unless directed", or
"when necessary") into condition. Otherwise set condition to null. Do not turn
a requirement, recommendation, permission, or prohibition into an observed
action. A prohibition is asserted with prohibited modality; status is negated
only when the evidence says the underlying relationship itself does not hold.
Do not
connect nearby entities, extract formatting, document containment, citations as
governance, templates, examples, specimen names, fill-in fields, or sentence
clauses as entities. Person means a named individual; use role or person_group
for non-individual actors. Return an empty assertions array when there is no
useful relationship.

Chunk:
${chunk.content}`,
		buildExtractionOutput(schema),
		4_000
	);
	return parseResult(result, chunk.content, 'llm');
}

export function selectEntityHints(
	text: string,
	schema: CorpusSchema,
	entities: ExtractedEntity[],
	limit = 40
): ExtractedEntity[] {
	const allowedTypes = new Set(schema.entityTypes.map((type) => type.name));
	const selected = entities
		.filter(
			(entity) =>
				allowedTypes.has(entity.type) &&
				Number.isInteger(entity.start) &&
				Number.isInteger(entity.end) &&
				entity.start >= 0 &&
				entity.end > entity.start &&
				entity.end <= text.length &&
				text.slice(entity.start, entity.end) === entity.text
		)
		.sort((left, right) => (right.score ?? 0) - (left.score ?? 0) || left.start - right.start);
	const uniqueHints = new Map<string, ExtractedEntity>();
	for (const entity of selected) {
		const key = `${entity.type}\u0000${entity.text}`;
		if (!uniqueHints.has(key)) uniqueHints.set(key, entity);
	}

	return [...uniqueHints.values()].slice(0, Math.max(0, limit));
}

export async function runGliner(
	chunks: GraphChunk[],
	schema: CorpusSchema,
	onResult?: (chunkId: string, result: ExtractionResult) => Promise<void>
): Promise<Map<string, ExtractionResult>> {
	if (!chunks.length) return new Map();
	const script = resolve(dirname(fileURLToPath(import.meta.url)), 'gliner-extractor.py');
	const python = process.env.KNOWLEDGE_GRAPH_PYTHON ?? process.env.PYTHON ?? pythonExecutable();
	const labels = schema.entityTypes.map((type) => type.name);
	if (process.env.KNOWLEDGE_GRAPH_GLINER_OTHER === 'true') {
		labels.push('other');
	}

	const byId = new Map(chunks.map((chunk) => [chunk.chunkId, chunk]));
	const results = new Map<string, ExtractionResult>();
	const child = spawn(python, [script], { stdio: ['pipe', 'pipe', 'pipe'] });
	let stderr = '';
	child.stderr.on('data', (data) => (stderr = `${stderr}${String(data)}`.slice(-12_000)));
	const exited = new Promise<number | null>((done, fail) => {
		child.on('error', fail);
		child.on('close', done);
	});
	child.stdin.end(
		JSON.stringify({
			chunks,
			entityTypes: labels,
			relationTypes: schema.relationTypes.map((type) => type.name)
		})
	);

	try {
		for await (const line of createInterface({ input: child.stdout })) {
			if (!line.trim()) continue;
			const item = JSON.parse(line) as Record<string, unknown>;
			const id = clean(item.chunkId);
			const chunk = byId.get(id);
			if (!chunk) continue;
			const result = parseResult(item, chunk.content, 'gliner');
			results.set(id, result);
			await onResult?.(id, result);
		}
	} catch (error) {
		child.kill();
		await exited.catch(() => undefined);
		throw error;
	}

	const code = await exited;
	if (code !== 0) {
		throw new Error(stderr.trim() || `GLiNER exited with code ${code}`);
	}
	return results;
}

export async function reconcileExtractions(
	text: string,
	schema: CorpusSchema,
	llm: ExtractionResult,
	gliner: ExtractionResult,
	settings: ExtractionSettings
): Promise<ExtractionResult> {
	const assertions = llm.assertions
		.filter((assertion) => isGroundedAssertion(text, assertion))
		.filter((assertion) => fitsSchema(assertion, schema))
		.filter((assertion) => isSemanticAssertionCandidate(text, assertion, schema))
		.map((assertion) => ({ ...assertion }));
	for (const candidate of gliner.assertions) {
		if (!isGroundedAssertion(text, candidate)) continue;
		if (!fitsSchema(candidate, schema)) continue;
		if (!isSemanticAssertionCandidate(text, candidate, schema)) continue;
		const match = assertions.find((assertion) => sameAssertion(assertion, candidate));
		if (match) {
			match.extractors = ['llm', 'gliner'];
			match.score = candidate.score;
			match.offsets ??= candidate.offsets;
		} else {
			assertions.push({ ...candidate });
		}
	}

	if (!assertions.length) return { assertions: [] };

	const accepted = await verify(text, assertions, schema, settings);

	return {
		assertions: assertions.flatMap((assertion) =>
			accepted.has(key(assertion)) ? [{ ...assertion, verified: true }] : []
		)
	};
}

export function emptyExtraction(): ExtractionResult {
	return { assertions: [] };
}

export function hasUsableText(text: string): boolean {
	const words = text.match(/\p{L}[\p{L}\p{N}'’-]{2,}/gu) ?? [];
	return words.length >= 3 && words.join('').length >= 12;
}

type VerificationDecision = {
	index: number;
	verdict: 'accept' | 'reverse' | 'reject';
	subjectTypeCorrect: boolean;
	objectTypeCorrect: boolean;
	modalityCorrect: boolean;
	conditionCorrect: boolean;
};

async function verify(
	text: string,
	assertions: ExtractedAssertion[],
	schema: CorpusSchema,
	settings: ExtractionSettings
): Promise<Set<string>> {
	const candidates = assertions.map((assertion, index) => ({
		index,
		subject: assertion.subject,
		subjectType: assertion.subjectType,
		predicate: assertion.rawPredicate,
		object: assertion.object,
		objectType: assertion.objectType,
		evidence: assertion.evidence,
		status: assertion.status,
		modality: assertion.modality,
		modalityCue: assertion.modalityCue,
		condition: assertion.condition
	}));

	const result = await askJson<{
		decisions: VerificationDecision[];
	}>(
		settings,
		`Strictly verify every candidate against the chunk and canonical schema.

Use "accept" only when:
- the evidence explicitly supports the relationship;
- the subject and object are in the proposed direction;
- both endpoint types are correct;
- the predicate accurately describes the relationship;
- modality preserves whether the text reports, requires, recommends, permits,
  prohibits, or describes a habitual responsibility;
- modalityCue is an exact substring that governs this relationship, is null for
  an unmarked observation, and does not confuse possibility with permission;
- condition is an exact substring containing any qualification that limits when
  the relationship applies, or null when it is unconditional.

Use "reverse" when the relationship is supported but its direction is reversed.
Use "reject" for co-occurrence, implication, vague wording, missing context,
incorrect predicates, unsupported types, unsupported claims, document-layout
provenance, citation-only evidence, templates, specimen examples, fill-in fields,
or sentence clauses used as entities.

Do not repair or rewrite candidates. Return exactly one decision for every
candidate index.

Canonical schema:
${JSON.stringify({
	entityTypes: promptEntityTypes(schema),
	relationTypes: schema.relationTypes.map(({ name, description, subjectTypes, objectTypes }) => ({
		name,
		description,
		subjectTypes,
		objectTypes
	}))
})}

Chunk:
${text}

Candidates:
${JSON.stringify(candidates)}`,
		VERIFICATION_OUTPUT,
		1_200
	);

	const acceptedIndexes = new Set(
		result.decisions
			.filter(
				(decision) =>
					Number.isInteger(decision.index) &&
					decision.verdict === 'accept' &&
					decision.subjectTypeCorrect &&
					decision.objectTypeCorrect &&
					decision.modalityCorrect &&
					decision.conditionCorrect
			)
			.map((decision) => decision.index)
	);

	return new Set(assertions.filter((_, index) => acceptedIndexes.has(index)).map(key));
}
function parseResult(
	value: Record<string, unknown>,
	text: string,
	extractor: Extractor
): ExtractionResult {
	const rawEntities = Array.isArray(value.entities) ? value.entities : [];
	const entities = rawEntities.flatMap((item) => {
		if (!item || typeof item !== 'object') return [];
		const row = item as Record<string, unknown>;
		const entityText = clean(row.text);
		const type = category(row.type ?? row.label);
		const start = integer(row.start);
		const end = integer(row.end);
		if (
			!entityText ||
			!type ||
			start === null ||
			end === null ||
			start < 0 ||
			end <= start ||
			end > text.length ||
			text.slice(start, end) !== entityText
		) {
			return [];
		}
		return [{ text: entityText, type, start, end, score: number(row.score) }];
	});
	const raw = Array.isArray(value.assertions) ? value.assertions : [];
	const assertions = raw.flatMap((item) => {
		if (!item || typeof item !== 'object') return [];
		const row = item as Record<string, unknown>;
		const subject = clean(row.subject);
		const object = clean(row.object);
		const rawPredicate = category(row.rawPredicate);
		const evidence = clean(row.evidence);
		if (
			!subject ||
			!object ||
			!rawPredicate ||
			normalize(subject) === normalize(object) ||
			!isGroundedAssertion(text, { subject, object, evidence })
		) {
			return [];
		}
		const status = clean(row.status);
		const modality = assertionModality(row.modality, evidence);
		const suppliedModalityCue = exactEvidenceValue(row.modalityCue, evidence);
		const modalityCue =
			suppliedModalityCue ??
			(extractor === 'gliner' ? inferredModalityCue(evidence, modality) : null);
		const condition = exactEvidenceValue(row.condition, evidence);
		const offsets = ['headStart', 'headEnd', 'tailStart', 'tailEnd'].map((name) =>
			integer(row[name])
		);
		return [
			{
				subject,
				subjectType: category(row.subjectType) || 'unknown',
				rawPredicate,
				object,
				objectType: category(row.objectType) || 'unknown',
				evidence,
				startDate: clean(row.startDate) || null,
				endDate: clean(row.endDate) || null,
				status: status === 'negated' || status === 'uncertain' ? status : 'asserted',
				modality,
				modalityCue,
				condition,
				extractors: [extractor],
				verified: false,
				score: number(row.score),
				offsets: offsets.every((value) => value !== null)
					? (offsets as [number, number, number, number])
					: null
			} satisfies ExtractedAssertion
		];
	});
	return {
		assertions: unique(assertions),
		entities: [
			...new Map(
				entities.map((entity) => [`${entity.type}\u0000${entity.start}\u0000${entity.end}`, entity])
			).values()
		]
	};
}

export function hasExactMention(text: string, value: string): boolean {
	if (!value) return false;
	let index = text.indexOf(value);

	while (index >= 0) {
		const before = [...text.slice(0, index)].at(-1) ?? '';
		const after = [...text.slice(index + value.length)][0] ?? '';
		const first = [...value][0] ?? '';
		const last = [...value].at(-1) ?? '';
		const startsAtBoundary = !isWordCharacter(first) || !isWordCharacter(before);
		const endsAtBoundary = !isWordCharacter(last) || !isWordCharacter(after);
		if (startsAtBoundary && endsAtBoundary) return true;
		index = text.indexOf(value, index + 1);
	}

	return false;
}

export function isGroundedAssertion(
	text: string,
	assertion: Pick<ExtractedAssertion, 'subject' | 'object' | 'evidence'>
): boolean {
	return (
		text.includes(assertion.evidence) &&
		hasExactMention(assertion.evidence, assertion.subject) &&
		hasExactMention(assertion.evidence, assertion.object)
	);
}

function isWordCharacter(value: string): boolean {
	return /[\p{L}\p{N}]/u.test(value);
}

function fitsSchema(assertion: ExtractedAssertion, schema: CorpusSchema): boolean {
	const relation = schema.relationTypes.find(
		(type) => type.name === category(assertion.rawPredicate)
	);
	if (!relation) return false;
	return relationAcceptsTypes(relation, assertion.subjectType, assertion.objectType);
}

async function askJson<T>(
	settings: ExtractionSettings,
	prompt: string,
	format: Record<string, unknown>,
	maxTokens: number,
	requestOptions: { thinking?: ProviderChatOptions['thinking'] } = {}
): Promise<T> {
	const attempts = positiveInteger(process.env.KNOWLEDGE_GRAPH_JSON_ATTEMPTS, 3);
	const contextWindow =
		settings.providerOptions?.contextSize ??
		positiveInteger(process.env.KNOWLEDGE_GRAPH_CONTEXT_WINDOW, DEFAULT_CONTEXT_WINDOW);
	const requestedMaxTokens = settings.providerOptions?.maxTokens ?? maxTokens;
	const structuredPrompt =
		`${prompt}\n\nReturn only JSON matching this schema:\n` + JSON.stringify(format);
	const estimatedInputTokens = estimateTokens(structuredPrompt);
	const availableOutputTokens =
		contextWindow - estimatedInputTokens - STRUCTURED_OUTPUT_CONTEXT_RESERVE;
	if (availableOutputTokens < 256) {
		throw new Error(
			`Structured prompt leaves only ${availableOutputTokens} output tokens in a ${contextWindow}-token context window.`
		);
	}
	const outputTokenLimit = Math.min(requestedMaxTokens, maxTokens, availableOutputTokens);
	const provider = getProvider(settings.providerId);

	let lastError: unknown;

	for (let attempt = 1; attempt <= attempts; attempt += 1) {
		let diagnostics = '';
		try {
			let output = '';
			let reasoningCharacters = 0;
			let finishReason: string | undefined;
			let actualInputTokens: number | undefined;
			let actualOutputTokens: number | undefined;
			const attemptPrompt =
				attempt === 1
					? structuredPrompt
					: `${structuredPrompt}\n\nThis is retry ${attempt}. Produce the JSON response immediately.`;

			for await (const chunk of provider.streamChat(
				[{ role: 'user', content: attemptPrompt }],
				settings.modelId,
				{
					...settings.providerOptions,
					temperature: 0,
					topK: 20,
					maxTokens: outputTokenLimit,
					contextSize: contextWindow,
					reasoningBudget: 0,
					thinking: attempt === attempts ? false : (requestOptions.thinking ?? false),
					structuredOutput: attempt === attempts ? undefined : format
				}
			)) {
				output += chunk.content ?? '';
				reasoningCharacters += chunk.reasoningContent?.length ?? 0;
				finishReason = chunk.finishReason ?? finishReason;
				actualInputTokens = chunk.inputTokens ?? actualInputTokens;
				actualOutputTokens = chunk.outputTokens ?? actualOutputTokens;
			}
			diagnostics = structuredDiagnostics({
				finishReason,
				inputTokens: actualInputTokens,
				outputTokens: actualOutputTokens,
				reasoningCharacters,
				contentCharacters: output.length,
				generationLimit: outputTokenLimit
			});

			return parseStructuredJson<T>(output);
		} catch (error) {
			lastError = new Error(`${message(error)}${diagnostics ? ` (${diagnostics})` : ''}`);

			if (attempt < attempts) {
				await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
			}
		}
	}

	throw new Error(
		`Structured model request failed after ${attempts} attempts: ${message(lastError)}`
	);
}

function structuredDiagnostics(options: {
	finishReason: string | undefined;
	inputTokens: number | undefined;
	outputTokens: number | undefined;
	reasoningCharacters: number;
	contentCharacters: number;
	generationLimit: number;
}): string {
	return [
		`finish=${options.finishReason ?? 'unknown'}`,
		`inputTokens=${options.inputTokens ?? 'unknown'}`,
		`outputTokens=${options.outputTokens ?? 'unknown'}`,
		`thinkingChars=${options.reasoningCharacters}`,
		`contentChars=${options.contentCharacters}`,
		`limit=${options.generationLimit}`
	].join(', ');
}

function parseStructuredJson<T>(output: string): T {
	const trimmed = output.trim();

	if (!trimmed) {
		throw new Error('Structured model returned an empty response.');
	}

	const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(trimmed);
	const json = fenced ? fenced[1].trim() : trimmed;

	return JSON.parse(json) as T;
}

async function loadOntologyTerms(sample: string): Promise<SchemaCategory[]> {
	try {
		const store = new Store();
		for (const url of ONTOLOGIES) {
			const response = await fetch(url);
			if (!response.ok) throw new Error(`${url} returned ${response.status}`);
			store.addQuads(new Parser({ baseIRI: url }).parse(await response.text()));
		}
		return selectOntologyTerms(store, sample);
	} catch (error) {
		console.warn(`Ontology loading failed; using universal seeds: ${error}`);
		return [];
	}
}

function selectOntologyTerms(store: Store, sample: string): SchemaCategory[] {
	const rdfType = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type';
	const kinds = [
		['http://www.w3.org/2000/01/rdf-schema#Class', false],
		['http://www.w3.org/2002/07/owl#Class', false],
		['http://www.w3.org/1999/02/22-rdf-syntax-ns#Property', true],
		['http://www.w3.org/2002/07/owl#ObjectProperty', true],
		['http://www.w3.org/2002/07/owl#DatatypeProperty', true]
	] as const;
	const sampleWords = new Set(words(sample));
	const found = new Map<string, ScoredTerm>();

	for (const [kind, relation] of kinds) {
		for (const quad of store.getQuads(null, rdfType, kind, null)) {
			if (quad.subject.termType !== 'NamedNode') continue;
			const id = quad.subject.value;
			const name = literal(store, id, 'label') ?? localName(id);
			const term = {
				name: category(name),
				description: literal(store, id, 'comment')?.slice(0, 300) ?? name,
				source: id.startsWith('https://schema.org/') ? 'schema.org' : 'prov-o',
				subjectTypes: relation ? objects(store, id, 'domain') : [],
				objectTypes: relation ? objects(store, id, 'range') : [],
				relation
			};
			const score = words(`${name} ${term.description}`).filter((word) =>
				sampleWords.has(word)
			).length;
			if (score) found.set(`${relation}:${term.name}`, { ...term, score });
		}
	}

	return [...found.values()]
		.sort((left, right) => right.score - left.score)
		.filter(
			(term, index, all) =>
				all.slice(0, index).filter((item) => item.relation === term.relation).length <
				(term.relation ? 30 : 20)
		)
		.map(({ name, description, source, subjectTypes, objectTypes }) => ({
			name,
			description,
			source,
			subjectTypes,
			objectTypes
		}));
}

function literal(store: Store, subject: string, name: 'label' | 'comment'): string | null {
	const predicate = `http://www.w3.org/2000/01/rdf-schema#${name}`;
	const values = store.getObjects(subject, predicate, null);
	return (
		values.find((value) => value.termType === 'Literal' && value.language === 'en')?.value ??
		values.find((value) => value.termType === 'Literal' && !value.language)?.value ??
		null
	);
}

function objects(store: Store, subject: string, name: 'domain' | 'range'): string[] {
	const predicate = `http://www.w3.org/2000/01/rdf-schema#${name}`;
	return store
		.getObjects(subject, predicate, null)
		.flatMap((value) => (value.termType === 'NamedNode' ? [category(localName(value.value))] : []));
}

function readCategories(value: unknown, relations = false): SchemaCategory[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((item) => {
		if (!item || typeof item !== 'object') return [];
		const row = item as Record<string, unknown>;
		const name = category(row.name);
		const description = clean(row.description);
		if (!name || !description || name === 'related_to') return [];
		return [
			{
				name,
				description,
				source: clean(row.source) || 'llm',
				...(relations
					? {
							subjectTypes: strings(row.subjectTypes).map(category),
							objectTypes: strings(row.objectTypes).map(category)
						}
					: {})
			}
		];
	});
}

// function sampleChunks(chunks: GraphChunk[], limit: number): GraphChunk[] {
// 	if (chunks.length <= limit) return chunks;
// 	const step = (chunks.length - 1) / (limit - 1);
// 	return [...new Set(Array.from({ length: limit }, (_, i) => chunks[Math.round(i * step)]))];
// }

function sameAssertion(left: ExtractedAssertion, right: ExtractedAssertion): boolean {
	if (
		normalize(left.subject) !== normalize(right.subject) ||
		normalize(left.object) !== normalize(right.object) ||
		left.modality !== right.modality ||
		normalize(left.condition ?? '') !== normalize(right.condition ?? '')
	) {
		return false;
	}
	const leftWords = new Set(words(left.rawPredicate));
	const rightWords = new Set(words(right.rawPredicate));
	const overlap = [...leftWords].filter((word) => rightWords.has(word)).length;
	return overlap > 0 && overlap / Math.min(leftWords.size, rightWords.size) >= 0.5;
}

function key(assertion: ExtractedAssertion): string {
	return [
		normalize(assertion.subject),
		category(assertion.rawPredicate),
		normalize(assertion.object),
		assertion.modality,
		assertion.condition ?? '',
		assertion.evidence
	].join('\u0000');
}

function unique<T extends { name?: string } | ExtractedAssertion>(items: T[]): T[] {
	return [
		...new Map(
			items.map((item) => ['subject' in item ? key(item) : (item.name ?? ''), item])
		).values()
	];
}

function pythonExecutable(): string {
	const local = resolve(
		process.cwd(),
		process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python'
	);
	if (existsSync(local)) return local;
	return process.platform === 'win32' ? 'python' : 'python3';
}

function words(value: string): string[] {
	return (
		value
			.replace(/([a-z])([A-Z])/g, '$1 $2')
			.toLowerCase()
			.match(/[a-z][a-z0-9]{2,}/g) ?? []
	);
}

function localName(value: string): string {
	return decodeURIComponent(value.split(/[/#]/).pop() ?? '');
}

function category(value: unknown): string {
	return clean(value)
		.replace(/([a-z])([A-Z])/g, '$1_$2')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '_')
		.replace(/^_+|_+$/g, '');
}

function strings(value: unknown): string[] {
	return Array.isArray(value) ? value.map(clean).filter(Boolean) : [];
}

function clean(value: unknown): string {
	return typeof value === 'string' ? value.trim() : '';
}

function normalize(value: string): string {
	return value
		.normalize('NFKC')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, ' ')
		.trim();
}

function integer(value: unknown): number | null {
	const parsed = Number(value);
	return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function number(value: unknown): number | null {
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

function assertionModality(value: unknown, evidence: string): AssertionModality {
	const parsed = category(value);
	if (
		parsed === 'observed' ||
		parsed === 'habitual' ||
		parsed === 'required' ||
		parsed === 'recommended' ||
		parsed === 'permitted' ||
		parsed === 'prohibited'
	) {
		return parsed;
	}
	if (/\b(?:must|shall)\s+not\b|\bprohibit(?:ed|s)?\b/i.test(evidence)) return 'prohibited';
	if (/\b(?:must|shall)\b|\b(?:mandatory|required)\b/i.test(evidence)) return 'required';
	if (/\bshould\b|\brecommend(?:ed|s)?\b/i.test(evidence)) return 'recommended';
	if (/\b(?:permitted|allowed|optional)\b|\bauthoriz(?:e|ed|es)\b/i.test(evidence)) {
		return 'permitted';
	}
	return 'observed';
}

function exactEvidenceValue(value: unknown, evidence: string): string | null {
	const candidate = clean(value);
	return candidate && evidence.includes(candidate) ? candidate : null;
}

function inferredModalityCue(evidence: string, modality: AssertionModality): string | null {
	const patterns: Partial<Record<AssertionModality, RegExp>> = {
		prohibited:
			/\b(?:must|shall|may)\s+not\b|\b(?:prohibited|forbidden)\b|\bnot\s+(?:permitted|authorized|allowed)\b/i,
		required:
			/\b(?:must|shall|will)\b|\b(?:mandatory|required)\b|\b(?:has|have|had)\s+to\b|\bneed(?:s|ed)?\s+to\b/i,
		recommended: /\bshould\b|\bought\s+to\b|\brecommend(?:ed|s)?\b/i,
		permitted: /\bmay\b|\b(?:permitted|allowed|optional)\b|\bauthoriz(?:e|ed|es)\b/i,
		habitual: /\b(?:usually|generally|routinely|regularly|customarily|typically)\b/i
	};
	return evidence.match(patterns[modality] ?? /$a/)?.[0] ?? null;
}

function positiveInteger(value: string | undefined, fallback: number): number {
	const parsed = Number(value);
	return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function boundedNumber(
	value: string | undefined,
	fallback: number,
	minimum: number,
	maximum: number
): number {
	if (value === undefined) return fallback;
	if (!value.trim()) {
		throw new Error(`The configured value must be between ${minimum} and ${maximum}.`);
	}
	const parsed = Number(value);
	if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) {
		throw new Error(`The configured value must be between ${minimum} and ${maximum}.`);
	}
	return parsed;
}

function message(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
