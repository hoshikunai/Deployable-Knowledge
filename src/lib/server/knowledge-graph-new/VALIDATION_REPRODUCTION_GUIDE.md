# Knowledge Graph Validation Reproduction Guide

This guide reproduces the validation described in
[Current Validation Report](./CURRENT_VALIDATION_REPORT.md) against the gates in
[Architecture and First-Run Findings](./ARCHITECTURE_AND_FINDINGS.md).

The procedure is intentionally read-only with respect to `app.db`. It directly
calls schema discovery and extraction functions instead of
`buildKnowledgeGraph()`, because the production build function creates and
updates `kg_new_*` tables.

## Expected Duration

On the test WSL machine with CPU-only `gemma4:latest`:

- Ollama cold load: approximately 2 minutes;
- diagnostic schema discovery: approximately 8-9 minutes;
- GLiNER over 30 chunks: approximately 80 seconds;
- warm LLM extraction: approximately 1.3-2.9 minutes per chunk;
- verification with candidates: approximately 1-3 minutes per chunk; and
- candidate-heavy retries: potentially several additional minutes.

A complete successful 30-chunk run can take multiple hours. The runner
checkpoints every completed stage and chunk so it can resume after interruption.

## 1. Start From the Correct Project and Branch

```bash
cd /home/cpach/Projects/Deployable-Knowledge
git branch --show-current
git status --short
```

Expected branch:

```text
kg-new
```

Do not discard or overwrite unrelated working-tree changes.

## 2. Verify Node Dependencies and Repository Health

```bash
node --version
npm --version
npm run lint
npm run check
npm run build
git diff --check
```

Required outcome:

- lint passes;
- `svelte-check` reports zero errors and zero warnings;
- the build passes; and
- `git diff --check` produces no output.

Run the build before creating temporary files in `.svelte-kit`, because a later
build may recreate that directory.

## 3. Verify the Python Environment

The test used `.venv` with Python 3.14. If the environment does not exist:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r src/lib/server/knowledge-graph-new/requirements.txt
```

If `uv` is installed, the dependency step may instead be run as:

```bash
uv pip install --python .venv/bin/python -r src/lib/server/knowledge-graph-new/requirements.txt
```

Verify the relevant packages:

```bash
.venv/bin/python -c "import gliner, pyarrow, torch; print({'gliner': gliner.__version__, 'pyarrow': pyarrow.__version__, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()})"
```

The tested environment reported:

```text
gliner: 0.2.27
pyarrow: 25.0.0
torch: 2.13.0+cpu
cuda: false
```

## 4. Verify Ollama and Freeze the Model Identity

Ensure Ollama is running and the expected model is installed:

```bash
node --input-type=module -e "const response=await fetch('http://127.0.0.1:11434/api/tags'); console.log(JSON.stringify(await response.json(),null,2));"
```

The tested model was:

```text
name: gemma4:latest
parameter size: 8.0B
quantization: Q4_K_M
digest: c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb
```

Record the digest. Results are not directly comparable if the tag points to a
different model digest.

## 5. Verify Native Structured Output and the 16K Context

Run a small native-schema request:

```bash
node --input-type=module -e "const schema={type:'object',additionalProperties:false,properties:{ok:{type:'boolean'}},required:['ok']}; const body={model:'gemma4:latest',messages:[{role:'user',content:'Return an object whose ok field is true. Output only the object.'}],format:schema,options:{temperature:0,top_k:20,num_predict:256,num_ctx:16384},stream:false,keep_alive:'30m'}; const response=await fetch('http://127.0.0.1:11434/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}); const payload=await response.json(); console.log(JSON.stringify({status:response.status,content:payload.message?.content,thinking:payload.message?.thinking,doneReason:payload.done_reason},null,2));"
```

Expected content:

```json
{ "ok": true }
```

While the model is loaded, verify its active context:

```bash
node --input-type=module -e "const response=await fetch('http://127.0.0.1:11434/api/ps'); console.log(JSON.stringify(await response.json(),null,2));"
```

Expected field:

```json
{ "context_length": 16384 }
```

## 6. Inventory the Stored Corpus

Run this read-only query:

```bash
node --input-type=module -e "import {createClient} from '@libsql/client'; const db=createClient({url:'file:app.db'}); const result=await db.execute(\"SELECT d.id,d.title,COUNT(c.id) AS chunks FROM documents d JOIN document_chunks c ON c.document_id=d.id WHERE c.chunk_type='TEXT' AND LENGTH(TRIM(c.content))>0 GROUP BY d.id,d.title ORDER BY d.title\"); console.log(JSON.stringify(result.rows,null,2)); db.close();"
```

The validation report used 5,291 nonempty text chunks across 11 documents.
Different input data will produce a different deterministic sample and cannot be
treated as an exact reproduction.

## 7. Compile the Current TypeScript Modules for a Temporary Runner

The runner imports bundled copies so it exercises the current working-tree code
without adding a permanent endpoint or test harness.

```bash
./node_modules/.bin/esbuild src/lib/server/knowledge-graph-new/extraction.ts --bundle --platform=node --format=esm --packages=external --tsconfig=tsconfig.json --outfile=.svelte-kit/kg-extraction-benchmark.mjs
./node_modules/.bin/esbuild src/lib/server/knowledge-graph-new/schema-sampling.ts --bundle --platform=node --format=esm --packages=external --tsconfig=tsconfig.json --outfile=.svelte-kit/kg-schema-sampling-benchmark.mjs
```

Do not run `npm run build` after this step until the benchmark is finished,
because the build may recreate `.svelte-kit`.

## 8. Create the Checkpointed Read-Only Runner

Save the following as:

```text
.svelte-kit/findings-benchmark-runner.mjs
```

```js
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import {
	discoverCorpusSchema,
	extractWithLlm,
	reconcileExtractions
} from './kg-extraction-benchmark.mjs';
import { buildSchemaSample } from './kg-schema-sampling-benchmark.mjs';

const outputPath = '/tmp/kg-current-findings-benchmark.json';
const settings = {
	providerId: 'ollama',
	modelId: 'gemma4:latest',
	providerOptions: { contextSize: 16_384 }
};

const state = existsSync(outputPath)
	? JSON.parse(readFileSync(outputPath, 'utf8'))
	: {
			startedAt: new Date().toISOString(),
			schema: null,
			selected: [],
			gliner: {},
			llm: {},
			final: {},
			errors: [],
			timings: {}
		};

function save() {
	state.updatedAt = new Date().toISOString();
	writeFileSync(outputPath, JSON.stringify(state, null, 2));
}

function log(event, details = {}) {
	console.log(JSON.stringify({ event, at: new Date().toISOString(), ...details }));
}

function message(error) {
	return error instanceof Error ? error.message : String(error);
}

async function unloadOllama() {
	const response = await fetch('http://127.0.0.1:11434/api/generate', {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({ model: settings.modelId, keep_alive: 0 })
	});
	if (!response.ok) throw new Error(`Ollama unload failed: ${response.status}`);
}

function parseGlinerAssertion(item) {
	return {
		subject: String(item.subject),
		subjectType: String(item.subjectType),
		rawPredicate: String(item.rawPredicate),
		object: String(item.object),
		objectType: String(item.objectType),
		evidence: String(item.evidence),
		startDate: item.startDate ?? null,
		endDate: item.endDate ?? null,
		status: item.status === 'negated' || item.status === 'uncertain' ? item.status : 'asserted',
		extractors: ['gliner'],
		verified: false,
		score: typeof item.score === 'number' ? item.score : null,
		offsets: [item.headStart, item.headEnd, item.tailStart, item.tailEnd].every(Number.isInteger)
			? [item.headStart, item.headEnd, item.tailStart, item.tailEnd]
			: null
	};
}

async function runGliner(chunks, schema) {
	const payload = {
		chunks,
		entityTypes: [...schema.entityTypes.map((type) => type.name), 'other'],
		relationTypes: schema.relationTypes.map((type) => type.name)
	};
	const child = spawn(
		'.venv/bin/python',
		['src/lib/server/knowledge-graph-new/gliner-extractor.py'],
		{ stdio: ['pipe', 'pipe', 'pipe'] }
	);
	let stdout = '';
	let stderr = '';
	child.stdout.on('data', (part) => (stdout += String(part)));
	child.stderr.on('data', (part) => (stderr += String(part)));
	child.stdin.end(JSON.stringify(payload));
	const code = await new Promise((resolve, reject) => {
		child.on('error', reject);
		child.on('close', resolve);
	});
	if (code !== 0) {
		throw new Error(stderr.trim() || `GLiNER exited with ${code}`);
	}
	return Object.fromEntries(
		stdout
			.trim()
			.split('\n')
			.filter(Boolean)
			.map((line) => {
				const row = JSON.parse(line);
				return [String(row.chunkId), { assertions: row.assertions.map(parseGlinerAssertion) }];
			})
	);
}

const db = createClient({ url: 'file:app.db' });
const rows = await db.execute(`
	SELECT c.id, c.document_id, c.content, c.page_index, c.chunk_index, d.title
	FROM document_chunks c
	JOIN documents d ON d.id = c.document_id
	WHERE c.chunk_type = 'TEXT' AND LENGTH(TRIM(c.content)) > 0
	ORDER BY c.document_id, c.page_index, c.chunk_index, c.id
`);
const allChunks = rows.rows.map((row) => ({
	chunkId: String(row.id),
	documentId: String(row.document_id),
	content: String(row.content),
	title: String(row.title),
	pageIndex: Number(row.page_index),
	chunkIndex: Number(row.chunk_index)
}));

if (!state.selected.length) {
	state.selected = buildSchemaSample(allChunks, {
		maxChunks: 30,
		maxCharacters: 1_000_000
	}).chunks;
	save();
}

log('sample-ready', {
	allChunks: allChunks.length,
	selected: state.selected.length,
	documents: new Set(state.selected.map((chunk) => chunk.documentId)).size
});

if (!state.schema) {
	const started = Date.now();
	log('schema-start');
	state.schema = await discoverCorpusSchema(allChunks, settings);
	state.timings.schemaMs = Date.now() - started;
	save();
	log('schema-complete', {
		elapsedMs: state.timings.schemaMs,
		entityTypes: state.schema.entityTypes.map((type) => type.name),
		relations: state.schema.relationTypes.map((type) => type.name)
	});
}

if (!Object.keys(state.gliner).length) {
	const started = Date.now();
	await unloadOllama();
	log('gliner-start');
	state.gliner = await runGliner(state.selected, state.schema);
	state.timings.glinerMs = Date.now() - started;
	save();
	log('gliner-complete', {
		elapsedMs: state.timings.glinerMs,
		assertions: Object.values(state.gliner).reduce(
			(total, result) => total + result.assertions.length,
			0
		)
	});
}

for (const [index, chunk] of state.selected.entries()) {
	if (!state.llm[chunk.chunkId]) {
		const started = Date.now();
		log('llm-start', {
			index: index + 1,
			chunkId: chunk.chunkId,
			title: chunk.title
		});
		try {
			state.llm[chunk.chunkId] = await extractWithLlm(chunk, state.schema, settings);
			state.timings[`llm:${chunk.chunkId}`] = Date.now() - started;
			save();
			log('llm-complete', {
				index: index + 1,
				chunkId: chunk.chunkId,
				elapsedMs: state.timings[`llm:${chunk.chunkId}`],
				assertions: state.llm[chunk.chunkId].assertions.length
			});
		} catch (error) {
			state.errors.push({
				stage: 'llm',
				chunkId: chunk.chunkId,
				message: message(error)
			});
			save();
			log('llm-error', { chunkId: chunk.chunkId, message: message(error) });
			continue;
		}
	}

	if (!state.final[chunk.chunkId]) {
		const started = Date.now();
		log('verify-start', { index: index + 1, chunkId: chunk.chunkId });
		try {
			state.final[chunk.chunkId] = await reconcileExtractions(
				chunk.content,
				state.schema,
				state.llm[chunk.chunkId],
				state.gliner[chunk.chunkId] ?? { assertions: [] },
				settings
			);
			state.timings[`verify:${chunk.chunkId}`] = Date.now() - started;
			save();
			log('verify-complete', {
				index: index + 1,
				chunkId: chunk.chunkId,
				elapsedMs: state.timings[`verify:${chunk.chunkId}`],
				assertions: state.final[chunk.chunkId].assertions.length
			});
		} catch (error) {
			state.errors.push({
				stage: 'verify',
				chunkId: chunk.chunkId,
				message: message(error)
			});
			save();
			log('verify-error', { chunkId: chunk.chunkId, message: message(error) });
		}
	}
}

state.completedAt = new Date().toISOString();
save();
log('benchmark-complete', {
	outputPath,
	errors: state.errors.length,
	llmAssertions: Object.values(state.llm).reduce(
		(total, result) => total + result.assertions.length,
		0
	),
	glinerAssertions: Object.values(state.gliner).reduce(
		(total, result) => total + result.assertions.length,
		0
	),
	finalAssertions: Object.values(state.final).reduce(
		(total, result) => total + result.assertions.length,
		0
	)
});
db.close();
```

Check its syntax:

```bash
node --check .svelte-kit/findings-benchmark-runner.mjs
```

## 9. Start With the Production Defaults

For a fresh run, ensure `/tmp/kg-current-findings-benchmark.json` does not contain
an older completed state. Preserve an existing file by moving it to a separate
explicit filename rather than deleting it.

Run the production configuration:

```bash
node --input-type=module -e "await import('./.svelte-kit/findings-benchmark-runner.mjs')"
```

In the reported environment, this failed during schema discovery after three
empty responses because the 30,000-character sample produced a 16,322-token
prompt inside a 16,384-token context.

Keep this failure as part of the validation record. Do not silently begin with a
smaller sample, because that would hide the production-default regression.

## 10. Run the 18K Diagnostic Override

After the production run exits, reuse its checkpoint and set only the schema
sample character limit:

```bash
node --input-type=module -e "process.env.KNOWLEDGE_GRAPH_SCHEMA_SAMPLE_CHARACTERS='18000'; await import('./.svelte-kit/findings-benchmark-runner.mjs')"
```

This is a diagnostic configuration, not the production result. Record that
distinction in any report.

The runner will:

1. reuse the already selected 30 chunks;
2. retry schema discovery with the smaller character limit;
3. checkpoint the schema;
4. unload Ollama to leave RAM for GLiNER;
5. run GLiNER once across all 30 chunks;
6. reload Ollama on the first LLM extraction;
7. extract and verify chunks sequentially; and
8. save after every successful or failed stage.

Pressing `Ctrl-C` is safe after a checkpoint event. Run the same command again to
resume. An interrupted in-flight verifier will run again because it has no final
checkpoint yet.

## 11. Monitor Token Counts and Progress

Runner progress is emitted as one JSON object per event:

```text
sample-ready
schema-start
schema-complete
gliner-start
gliner-complete
llm-start
llm-complete
llm-error
verify-start
verify-complete
verify-error
benchmark-complete
```

If Ollama runs as a systemd service, inspect prompt and generation progress:

```bash
journalctl -u ollama --since '10 minutes ago' --no-pager | rg 'new prompt|n_ctx_slot|task.n_tokens|init_sampler|n_decoded|print_timing'
```

The important line resembles:

```text
new prompt, n_ctx_slot = 16384, task.n_tokens = 16322
```

Interpret it as:

```text
context window = n_ctx_slot
complete input prompt = task.n_tokens
```

Calculate the request budget as:

```text
required context = prompt tokens + maximum output tokens + safety margin
```

Recommended safety margin for this test: at least 1,024 tokens.

Stage output limits in the current code are:

```text
schema discovery: 2,000
chunk extraction: 4,000
verification:     1,200
```

## 12. Inspect the Checkpoint

The checkpoint is:

```text
/tmp/kg-current-findings-benchmark.json
```

Print high-level counts:

```bash
node --input-type=module -e "import {readFileSync} from 'node:fs'; const state=JSON.parse(readFileSync('/tmp/kg-current-findings-benchmark.json','utf8')); const count=(records)=>Object.values(records).reduce((total,result)=>total+(result.assertions?.length??0),0); console.log(JSON.stringify({selected:state.selected.length,schema:state.schema,glinerChunks:Object.keys(state.gliner).length,llmChunks:Object.keys(state.llm).length,finalChunks:Object.keys(state.final).length,glinerAssertions:count(state.gliner),llmAssertions:count(state.llm),finalAssertions:count(state.final),errors:state.errors,timings:state.timings},null,2));"
```

Print every final assertion with its source document and chunk:

```bash
node --input-type=module -e "import {readFileSync} from 'node:fs'; const state=JSON.parse(readFileSync('/tmp/kg-current-findings-benchmark.json','utf8')); const chunks=new Map(state.selected.map((chunk)=>[chunk.chunkId,chunk])); const rows=Object.entries(state.final).flatMap(([chunkId,result])=>result.assertions.map((assertion)=>({chunkId,title:chunks.get(chunkId)?.title,pageIndex:chunks.get(chunkId)?.pageIndex,...assertion}))); console.log(JSON.stringify(rows,null,2));"
```

Print per-chunk raw and final counts:

```bash
node --input-type=module -e "import {readFileSync} from 'node:fs'; const state=JSON.parse(readFileSync('/tmp/kg-current-findings-benchmark.json','utf8')); console.log(JSON.stringify(Object.keys(state.llm).map((chunkId,index)=>{const chunk=state.selected.find((item)=>item.chunkId===chunkId); return {index:index+1,chunkId,title:chunk?.title,pageIndex:chunk?.pageIndex,llm:state.llm[chunkId]?.assertions.length??0,gliner:state.gliner[chunkId]?.assertions.length??0,final:state.final[chunkId]?.assertions.length??null,error:state.errors.find((item)=>item.chunkId===chunkId)?.message??null};}),null,2));"
```

## 13. Verify Mechanical Grounding

Every retained assertion must satisfy all of these conditions:

- evidence is an exact substring of the selected chunk;
- evidence contains the exact subject;
- evidence contains the exact object; and
- subject and object differ after normalization.

Run:

```bash
node --input-type=module -e "import {readFileSync} from 'node:fs'; const state=JSON.parse(readFileSync('/tmp/kg-current-findings-benchmark.json','utf8')); const chunks=new Map(state.selected.map((chunk)=>[chunk.chunkId,chunk])); const rows=Object.entries(state.final).flatMap(([chunkId,result])=>result.assertions.map((assertion)=>{const text=chunks.get(chunkId)?.content??''; return {chunkId,subject:assertion.subject,predicate:assertion.rawPredicate,object:assertion.object,evidenceExact:text.includes(assertion.evidence),subjectInEvidence:assertion.evidence.includes(assertion.subject),objectInEvidence:assertion.evidence.includes(assertion.object)};})); console.log(JSON.stringify(rows,null,2)); if(rows.some((row)=>!row.evidenceExact||!row.subjectInEvidence||!row.objectInEvidence)) process.exitCode=1;"
```

## 14. Check Schema Type Consistency

Every relation endpoint type should exist in the entity vocabulary:

```bash
node --input-type=module -e "import {readFileSync} from 'node:fs'; const state=JSON.parse(readFileSync('/tmp/kg-current-findings-benchmark.json','utf8')); const names=new Set(state.schema.entityTypes.map((type)=>type.name)); const rows=state.schema.relationTypes.map((relation)=>({relation:relation.name,missingSubjectTypes:(relation.subjectTypes??[]).filter((type)=>!names.has(type)),missingObjectTypes:(relation.objectTypes??[]).filter((type)=>!names.has(type))})); console.log(JSON.stringify(rows,null,2)); if(rows.some((row)=>row.missingSubjectTypes.length||row.missingObjectTypes.length)) process.exitCode=1;"
```

The reported diagnostic schema fails this check for `protocol` and
`military_unit`.

## 15. Perform the Manual Quality Review

Automated grounding is not a semantic quality label. Review every retained
assertion against its complete chunk and record:

- `useful`: the complete triple is a defensible, durable graph fact;
- `directionCorrect`: the subject and object are in the source-supported
  direction;
- `endpointTypesCorrect`: both endpoint types accurately describe the endpoint;
- `canonicalRelation`: the mapped relation name, or `null` if no canonical
  relation accurately represents the source; and
- source category: `llm-only`, `gliner-only`, or `agreement`.

Reject as not useful when an assertion is primarily:

- a page header, page number, table, appendix, heading, inventory layout, or
  document containment fact;
- a sentence or verb fragment used as an entity;
- co-occurrence without an explicit relationship;
- a reversed or forced relation;
- a recommendation, prohibition, permission, failure, or condition flattened
  into an ordinary fact; or
- a relation whose canonical predicate changes the source meaning.

Use `calculateQualityMetrics()` from `quality-metrics.ts` after creating a
review object with the interfaces defined in that file. Recall requires a
separate manually authored expected-assertion inventory for the selected chunks;
do not invent an expected count from model output.

The required gates are:

```text
useful-triple precision:       >= 85%
direction accuracy:            >= 90%
endpoint-type accuracy:        >= 90%
canonical-relation coverage:   >= 90%
```

## 16. Compare Extractors Separately

Report final assertions by provenance:

```bash
node --input-type=module -e "import {readFileSync} from 'node:fs'; const state=JSON.parse(readFileSync('/tmp/kg-current-findings-benchmark.json','utf8')); const assertions=Object.values(state.final).flatMap((result)=>result.assertions); const counts={llmOnly:assertions.filter((item)=>item.extractors.length===1&&item.extractors[0]==='llm').length,glinerOnly:assertions.filter((item)=>item.extractors.length===1&&item.extractors[0]==='gliner').length,agreement:assertions.filter((item)=>item.extractors.length===2).length}; console.log(JSON.stringify(counts,null,2));"
```

Also report raw candidate counts. A small final GLiNER count can mean either low
GLiNER recall or aggressive reconciliation; it cannot be interpreted without
the raw count and verifier failures.

## 17. Verify That the Test Did Not Mutate the Database

```bash
node --input-type=module -e "import {createClient} from '@libsql/client'; const db=createClient({url:'file:app.db'}); const result=await db.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kg_new_%' ORDER BY name\"); console.log(JSON.stringify(result.rows,null,2)); db.close();"
```

For the read-only validation, the result should remain unchanged from its
pre-test value. In the reported run, it was an empty array before and after.

## 18. Report Required Limitations

Every report produced from this procedure must state:

- whether production defaults or a diagnostic override were used;
- model name and digest;
- context size and observed prompt token count;
- corpus chunk and document counts;
- selected, extracted, finalized, and failed chunk counts;
- whether the 30-50 chunk gate completed;
- raw LLM and GLiNER candidate counts;
- final provenance counts;
- grounding and manual quality metrics;
- whether recall had a complete expected-assertion label set; and
- any stopped, retried, or interrupted model calls.

Do not report a partial diagnostic run as a successful 30-chunk benchmark.
