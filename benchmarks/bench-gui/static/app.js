const byId = (id) => document.getElementById(id);

// [key, label, hint, minimum]; hints marked live are recomputed as values change
const SETTINGS_GROUPS = [
	{
		title: 'Retrieval depths',
		fields: [
			['documentDepth', 'Documents scored per query', 'live', 10],
			['retrievalMultiplier', 'Prefetch multiplier', 'live', 1],
			['rerankMultiplier', 'Rerank shortlist multiplier', 'live', 1],
			['rrfRankConstant', 'RRF k', 'Rank constant in 1 / (k + rank)', 1]
		]
	},
	{
		title: 'Reranker',
		fields: [
			[
				'rerankMaxTokens',
				'Reranker max tokens',
				'Query + passage tokens read before truncating. Ettin up to 7,999; MS MARCO up to 512.',
				16
			]
		]
	},
	{
		title: 'Corpus chunking',
		fields: [
			['chunkMaxTokens', 'Chunk size (tokens)', 'live', 0],
			['chunkOverlapTokens', 'Chunk overlap (tokens)', 'live', 0]
		]
	},
	{
		title: 'Sessions and system',
		fields: [
			['timeoutMinutes', 'Session timeout (minutes)', 'Runs resume automatically after it', 1],
			['maxSessions', 'Max sessions per run', 'Stops and waits for Resume after this', 1],
			['memoryCeilingGb', 'Memory ceiling (GB)', 'Pauses the run above this', 1],
			['appPort', 'App server port', 'Used while benchmarking', 1024],
			['guiPort', 'GUI port', 'Takes effect next launch', 1024]
		]
	}
];
const SETTINGS_FIELDS = SETTINGS_GROUPS.flatMap((group) => group.fields);
const RESUMABLE = ['memory-paused', 'failed', 'cancelled', 'interrupted', 'sessions-exhausted'];
const BUILD_ITEM = '__build__';
const LOG_LINES = 2000;

// Characters per embedding token, measured on the prepared SciFact and NFCorpus (4.45) and
// ArguAna (4.97) chunks. Real text varies, so hints show the range rather than one number.
const CHARACTERS_PER_TOKEN = { low: 4.45, high: 4.97 };
const DEFAULT_CHUNK_CHARACTERS = 1200;

let options = null;
let runs = [];
const selectedRuns = new Set();
let lastScores = null;

async function api(path, init = {}) {
	const response = await fetch(path, {
		headers: { 'Content-Type': 'application/json' },
		...init
	});
	const body = await response.json().catch(() => ({}));
	if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`);
	return body;
}

function escapeHtml(value) {
	return String(value)
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;');
}

function formatTime(seconds) {
	if (!seconds) return 'never';
	return new Date(seconds * 1000).toLocaleString(undefined, {
		month: 'short',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit'
	});
}

// Tabs

function selectTab(name) {
	for (const tab of document.querySelectorAll('.tab')) {
		const selected = tab.dataset.tab === name;
		tab.setAttribute('aria-selected', String(selected));
		byId(`tab-${tab.dataset.tab}`).hidden = !selected;
	}
	if (name === 'runs') loadRuns();
}

// Run form

function selectedPipeline() {
	return document.querySelector('input[name="pipeline"]:checked')?.value ?? 'rrf';
}

function describeDepths() {
	const settings = options.settings;
	const pipeline = options.pipelines.find((item) => item.key === selectedPipeline());
	const output = settings.documentDepth * 2;
	const prefetch = output * settings.retrievalMultiplier;
	const steps = [
		`Semantic + BM25 fetch ${prefetch} chunks each`,
		`RRF (k=${settings.rrfRankConstant})`
	];
	if (pipeline.reranker) {
		steps.push(
			`${pipeline.label.replace('RRF + ', '')} reranks ${output * settings.rerankMultiplier}`
		);
	}
	steps.push(`${output} chunks → top ${settings.documentDepth} documents scored`);
	byId('depth-summary').textContent = `${steps.join(' → ')}. Change depths in Settings.`;
}

function renderOptions() {
	byId('corpora-legend').textContent = `Corpora · ${options.corpora[0]?.chunking ?? ''}`;
	byId('corpora').innerHTML = options.corpora
		.map((corpus) => {
			const detail = corpus.prepared
				? `${corpus.queryCount ?? '?'} queries · ${corpus.corpusSize} docs`
				: 'not prepared';
			return `<label class="choice ${corpus.prepared ? '' : 'disabled'}">
				<input type="checkbox" name="corpus" value="${corpus.name}" ${corpus.prepared ? '' : 'disabled'}>
				<span><strong>${corpus.name}</strong><br><span class="hint">${escapeHtml(corpus.description)}</span></span>
				<small>${escapeHtml(detail)}${corpus.prepared ? '' : `<br>${escapeHtml(corpus.reason)}. Prepare it in Settings.`}</small>
			</label>`;
		})
		.join('');
	byId('pipelines').innerHTML = options.pipelines
		.map(
			(pipeline, index) => `<label class="choice">
				<input type="radio" name="pipeline" value="${pipeline.key}" ${index === 0 ? 'checked' : ''}>
				${escapeHtml(pipeline.label)}
			</label>`
		)
		.join('');
	const filter = byId('runs-filter');
	filter.innerHTML =
		'<option value="">All</option>' +
		options.corpora.map((corpus) => `<option>${corpus.name}</option>`).join('');
	describeDepths();
}

function updateQueryMode() {
	const sample = document.querySelector('input[name="query-mode"]:checked').value === 'sample';
	byId('sample-size').disabled = !sample;
	byId('seeds-row').hidden = !sample;
}

async function startJobs(event) {
	event.preventDefault();
	const datasets = [...document.querySelectorAll('input[name="corpus"]:checked')].map(
		(input) => input.value
	);
	const sample = document.querySelector('input[name="query-mode"]:checked').value === 'sample';
	byId('job-error').textContent = '';
	try {
		await api('/api/jobs', {
			method: 'POST',
			body: JSON.stringify({
				datasets,
				pipeline: selectedPipeline(),
				queries: sample ? byId('sample-size').value : 'all',
				seeds: sample ? byId('seeds').value : '42'
			})
		});
		await refreshState();
	} catch (error) {
		byId('job-error').textContent = error.message;
	}
}

// Current job, queue, and log

function jobStatusLine(job) {
	if (job.kind === 'prepare') {
		return `${job.dataset} · preparing ${job.settings.chunkMaxTokens}-token chunks, ${job.settings.chunkOverlapTokens} overlap`;
	}
	const pipeline = options?.pipelines.find((item) => item.key === job.pipeline)?.label ?? '';
	const queries =
		job.queries === 'all' ? 'all queries' : `${job.queries} queries, seed ${job.seed}`;
	return `${job.dataset} · ${pipeline} · ${queries}`;
}

function renderCurrent(current) {
	const container = byId('current');
	if (!current) {
		container.textContent = 'Idle.';
		return;
	}
	if (current === BUILD_ITEM) {
		container.innerHTML = '<p>Building the app…</p>';
		return;
	}
	const percent = current.total ? Math.round((100 * current.completed) / current.total) : 0;
	container.innerHTML = `
		<p><strong>${escapeHtml(jobStatusLine(current))}</strong></p>
		<p class="mono">${escapeHtml(current.runId)}</p>
		<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}"><div style="width:${percent}%"></div></div>
		<p class="hint">${current.completed} / ${current.total ?? '?'} ${current.kind === 'prepare' ? 'documents ingested' : `queries · session ${Math.max(current.sessions, 1)}`}</p>
		<button type="button" class="button small" id="cancel-job">Cancel</button>`;
	byId('cancel-job').addEventListener('click', cancelJob);
}

function renderQueue(queue) {
	const list = byId('queue');
	if (queue.length === 0) {
		list.innerHTML = '<li class="hint">Empty.</li>';
		return;
	}
	list.innerHTML = queue
		.map((job) =>
			job.runId === BUILD_ITEM
				? '<li><span>App rebuild</span></li>'
				: `<li><button type="button" class="button small" data-dequeue="${job.runId}" aria-label="Remove ${job.runId} from queue">Remove</button><span>${escapeHtml(job.runId)}</span></li>`
		)
		.join('');
}

async function refreshState() {
	try {
		const state = await api('/api/state');
		renderCurrent(state.current);
		renderQueue(state.queue);
	} catch (error) {
		byId('current').textContent = `GUI server unreachable: ${error.message}`;
	}
}

async function cancelJob() {
	try {
		await api('/api/jobs/cancel', { method: 'POST' });
	} catch (error) {
		appendLog(`Cancel failed: ${error.message}`);
	}
}

async function dequeue(runId) {
	await api(`/api/jobs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
	await refreshState();
}

function appendLog(text) {
	const log = byId('log');
	const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
	log.append(`${text}\n`);
	while (log.childNodes.length > LOG_LINES) log.firstChild.remove();
	if (atBottom) log.scrollTop = log.scrollHeight;
	if (/finished \(exit/.test(text)) {
		loadBuild();
		if (!byId('tab-runs').hidden) loadRuns();
	}
}

function connectLog() {
	let last = -1;
	const source = new EventSource('/api/events?after=0');
	source.onmessage = (event) => {
		const line = JSON.parse(event.data);
		if (line.seq <= last) return;
		last = line.seq;
		appendLog(line.text);
	};
}

// Build status

async function loadBuild() {
	const build = await api('/api/build');
	const status = byId('build-status');
	status.classList.toggle('stale', build.stale);
	const dirty = build.sourceDirty ? ' + uncommitted src changes' : '';
	status.textContent = build.stale
		? `App build (${formatTime(build.buildTime)}) is older than src/ — rebuild before benchmarking`
		: `App build ${formatTime(build.buildTime)} · ${build.sourceCommit.slice(0, 7)}${dirty}`;
	byId('rebuild').hidden = false;
	byId('rebuild').classList.toggle('primary', build.stale);
}

async function rebuild() {
	try {
		await api('/api/build', { method: 'POST' });
		selectTab('run');
		await refreshState();
	} catch (error) {
		appendLog(`Rebuild failed to queue: ${error.message}`);
	}
}

// Runs

function runStatus(run) {
	if (run.jobState && run.jobState !== 'complete' && run.status !== 'complete') {
		return run.jobState;
	}
	return run.status;
}

function renderRuns() {
	const filter = byId('runs-filter').value;
	const rows = runs.filter((run) => !filter || run.dataset === filter);
	byId('runs-body').innerHTML =
		rows
			.map((run) => {
				const status = runStatus(run);
				const ndcg = run.ndcg10.hybrid;
				const queries =
					run.selection === 'seeded-shuffle'
						? `${run.queryCount} (seed ${run.sampleSeed})`
						: `${run.queryCount} (all)`;
				const resume =
					run.resumable && RESUMABLE.includes(run.jobState) && run.status !== 'complete'
						? `<button type="button" class="button small" data-resume="${run.id}">Resume</button>`
						: '';
				const progress = status === 'complete' ? '' : ` ${run.completed}/${run.queryCount}`;
				return `<tr>
					<td><input type="checkbox" data-select="${run.id}" aria-label="Select ${run.id}" ${selectedRuns.has(run.id) ? 'checked' : ''}></td>
					<td class="mono">${escapeHtml(run.id)}</td>
					<td>${escapeHtml(run.pipelineLabel)}<br><span class="hint">${escapeHtml(run.corpusLabel)}${run.hasScores ? '' : ' · no chunk scores'}</span></td>
					<td>${queries}</td>
					<td class="status-${status}">${escapeHtml(status)}${progress}${run.jobMessage && status !== 'complete' ? `<br><span class="hint">${escapeHtml(run.jobMessage)}</span>` : ''}</td>
					<td class="numeric">${ndcg == null ? '–' : ndcg.toFixed(5)}</td>
					<td><div class="actions">
						<button type="button" class="button small" data-scores="${run.id}">Scores</button>
						${resume}
						<button type="button" class="button small" data-delete="${run.id}">Delete</button>
					</div></td>
				</tr>`;
			})
			.join('') || '<tr><td colspan="7" class="hint">No runs yet.</td></tr>';
	byId('score-selected').disabled = selectedRuns.size === 0;
}

async function loadRuns() {
	runs = await api('/api/runs');
	for (const id of [...selectedRuns]) {
		if (!runs.some((run) => run.id === id)) selectedRuns.delete(id);
	}
	renderRuns();
}

function confirmDialog(title, message) {
	const dialog = byId('confirm-dialog');
	byId('confirm-title').textContent = title;
	byId('confirm-message').textContent = message;
	dialog.returnValue = 'cancel';
	dialog.showModal();
	return new Promise((resolve) => {
		dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), {
			once: true
		});
	});
}

async function deleteRun(runId) {
	const confirmed = await confirmDialog(
		'Delete run?',
		`${runId} and its supervisor files will be permanently deleted.`
	);
	if (!confirmed) return;
	try {
		await api(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
		selectedRuns.delete(runId);
	} catch (error) {
		appendLog(`Delete failed: ${error.message}`);
	}
	await loadRuns();
}

async function resumeRun(runId) {
	try {
		await api(`/api/runs/${encodeURIComponent(runId)}/resume`, { method: 'POST' });
		selectTab('run');
		await refreshState();
	} catch (error) {
		appendLog(`Resume failed: ${error.message}`);
	}
}

function handleRunsClick(event) {
	const target = event.target.closest('button');
	if (!target) return;
	if (target.dataset.scores) showScores([target.dataset.scores]);
	if (target.dataset.delete) deleteRun(target.dataset.delete);
	if (target.dataset.resume) resumeRun(target.dataset.resume);
}

function handleRunSelection(event) {
	const id = event.target.dataset.select;
	if (!id) return;
	if (event.target.checked) selectedRuns.add(id);
	else selectedRuns.delete(id);
	byId('score-selected').disabled = selectedRuns.size === 0;
}

// Scores

function inlineMarkdown(text) {
	return escapeHtml(text)
		.replace(/`([^`]+)`/g, '<code>$1</code>')
		.replace(/^_(.+)_$/, '<em>$1</em>');
}

function tableHtml(lines) {
	const cells = (line) =>
		line
			.replace(/^\||\|$/g, '')
			.split('|')
			.map((cell) => cell.trim());
	const [header, , ...body] = lines;
	return `<div class="table-wrap"><table><thead><tr>${cells(header)
		.map((cell) => `<th>${inlineMarkdown(cell)}</th>`)
		.join('')}</tr></thead><tbody>${body
		.map(
			(line) =>
				`<tr>${cells(line)
					.map((cell) => `<td>${inlineMarkdown(cell)}</td>`)
					.join('')}</tr>`
		)
		.join('')}</tbody></table></div>`;
}

// Renders the subset of Markdown the scores endpoint produces: headings, tables, paragraphs.
function renderMarkdown(markdown) {
	const html = [];
	const lines = markdown.split('\n');
	for (let index = 0; index < lines.length; index += 1) {
		const line = lines[index];
		if (line.startsWith('|')) {
			const table = [];
			while (index < lines.length && lines[index].startsWith('|')) table.push(lines[index++]);
			index -= 1;
			html.push(tableHtml(table));
		} else if (line.startsWith('### ')) {
			html.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`);
		} else if (line.startsWith('## ')) {
			html.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`);
		} else if (line.trim()) {
			html.push(`<p>${inlineMarkdown(line)}</p>`);
		}
	}
	return html.join('');
}

async function showScores(runIds) {
	selectTab('scores');
	byId('scores').innerHTML = '<p class="hint">Computing scores…</p>';
	byId('scores-title').textContent = runIds.join(', ');
	byId('copy-status').textContent = '';
	try {
		lastScores = await api(`/api/scores?runs=${encodeURIComponent(runIds.join(','))}`);
		byId('scores').innerHTML = renderMarkdown(lastScores.markdown);
	} catch (error) {
		lastScores = null;
		byId('scores').innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
	}
	byId('copy-json').disabled = lastScores === null;
	byId('copy-markdown').disabled = lastScores === null;
}

async function copyText(text, label) {
	try {
		await navigator.clipboard.writeText(text);
		byId('copy-status').textContent = `${label} copied.`;
	} catch {
		const area = document.createElement('textarea');
		area.value = text;
		document.body.append(area);
		area.select();
		const copied = document.execCommand('copy');
		area.remove();
		byId('copy-status').textContent = copied ? `${label} copied.` : 'Copy failed.';
	}
}

// Settings

function settingsField([key, label, hint, minimum], values) {
	const step = key === 'memoryCeilingGb' ? '0.5' : '1';
	const help = hint === 'live' ? '' : escapeHtml(hint);
	return `<label>${label}
		<input class="field" type="number" name="${key}" value="${values[key]}" step="${step}" min="${minimum}">
		<span class="hint" data-hint="${key}">${help}</span>
	</label>`;
}

function renderSettingsFields(values) {
	byId('settings-fields').innerHTML = SETTINGS_GROUPS.map(
		(group) => `<fieldset class="settings-group">
			<legend>${group.title}</legend>
			<div class="settings-grid">${group.fields.map((field) => settingsField(field, values)).join('')}</div>
			${group.title === 'Corpus chunking' ? '<div id="corpus-copies" class="corpus-copies"></div>' : ''}
		</fieldset>`
	).join('');
	updateLiveHints();
	loadCorpusCopies();
}

function formSettings() {
	const form = new FormData(byId('settings-form'));
	return Object.fromEntries(SETTINGS_FIELDS.map(([key]) => [key, Number(form.get(key))]));
}

function approximateCharacters(tokens) {
	const low = Math.round((tokens * CHARACTERS_PER_TOKEN.low) / 10) * 10;
	const high = Math.round((tokens * CHARACTERS_PER_TOKEN.high) / 10) * 10;
	return `≈ ${low.toLocaleString()}–${high.toLocaleString()} characters`;
}

function chunkSizeHint(tokens) {
	if (!tokens) {
		const low = Math.round(DEFAULT_CHUNK_CHARACTERS / CHARACTERS_PER_TOKEN.high);
		const high = Math.round(DEFAULT_CHUNK_CHARACTERS / CHARACTERS_PER_TOKEN.low);
		return `0 = default 1,200-character chunks (≈ ${low}–${high} tokens)`;
	}
	return `${approximateCharacters(tokens)} (the default is 1,200 characters)`;
}

function chunkOverlapHint(tokens, chunkTokens) {
	if (!chunkTokens) return 'Used with a token chunk size; the default overlaps one sentence';
	if (!tokens) return 'No overlap between neighbouring chunks';
	return `${approximateCharacters(tokens)}, carried back as whole sentences`;
}

// Shows the Top K each multiplier produces, since the multipliers scale the output chunk count
function updateLiveHints() {
	const values = formSettings();
	const output = values.documentDepth * 2;
	const hints = {
		documentDepth: `= ${output} output chunks per query (×2 so ${values.documentDepth} unique documents remain)`,
		retrievalMultiplier: `= ${output * values.retrievalMultiplier} chunks fetched by semantic and by BM25`,
		rerankMultiplier: `= ${output * values.rerankMultiplier} RRF chunks reranked (reranker pipelines only)`,
		chunkMaxTokens: chunkSizeHint(values.chunkMaxTokens),
		chunkOverlapTokens: chunkOverlapHint(values.chunkOverlapTokens, values.chunkMaxTokens)
	};
	for (const [key, text] of Object.entries(hints)) {
		const hint = document.querySelector(`[data-hint="${key}"]`);
		if (hint) hint.textContent = text.includes('NaN') ? '' : text;
	}
}

let copiesRequest = 0;

async function loadCorpusCopies() {
	const container = byId('corpus-copies');
	if (!container) return;
	const { chunkMaxTokens, chunkOverlapTokens } = formSettings();
	if (!chunkMaxTokens) {
		container.innerHTML =
			'<p class="hint">Default copies (1,200-character chunks) are used. Enter a chunk size in tokens to prepare other copies.</p>';
		return;
	}
	const request = ++copiesRequest;
	const params = new URLSearchParams({ chunkMaxTokens, chunkOverlapTokens });
	try {
		const corpora = await api(`/api/corpora?${params}`);
		if (request !== copiesRequest) return;
		container.innerHTML = `<p class="hint">Copies for ${escapeHtml(corpora[0].chunking)}. Preparing ingests the whole corpus and can take a long time.</p>
			<ul class="plain-list">${corpora
				.map((corpus) => {
					const action =
						corpus.state === 'prepared' || corpus.reason === 'BEIR dataset files are not downloaded'
							? ''
							: `<button type="button" class="button small" data-prepare="${corpus.name}">Save and prepare</button>`;
					const detail =
						corpus.state === 'prepared'
							? `prepared · ${corpus.documentCount} documents, ${corpus.chunkCount ?? '?'} chunks`
							: `${corpus.state} · ${corpus.reason}`;
					return `<li><strong>${corpus.name}</strong><span class="hint">${escapeHtml(detail)}</span>${action}</li>`;
				})
				.join('')}</ul>`;
	} catch (error) {
		if (request === copiesRequest) {
			container.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
		}
	}
}

async function persistSettings() {
	options.settings = await api('/api/settings', {
		method: 'PUT',
		body: JSON.stringify(formSettings())
	});
	options = await api('/api/options');
	renderOptions();
}

function openSettings() {
	renderSettingsFields(options.settings);
	byId('settings-error').textContent = '';
	byId('settings-dialog').showModal();
}

async function saveSettings(event) {
	event.preventDefault();
	try {
		await persistSettings();
		byId('settings-dialog').close();
	} catch (error) {
		byId('settings-error').textContent = error.message;
	}
}

async function prepareCorpus(dataset) {
	byId('settings-error').textContent = '';
	try {
		await persistSettings();
		await api('/api/corpora/prepare', {
			method: 'POST',
			body: JSON.stringify({ datasets: [dataset] })
		});
		byId('settings-dialog').close();
		selectTab('run');
		await refreshState();
	} catch (error) {
		byId('settings-error').textContent = error.message;
	}
}

// Startup

function bindEvents() {
	for (const tab of document.querySelectorAll('.tab')) {
		tab.addEventListener('click', () => selectTab(tab.dataset.tab));
	}
	byId('job-form').addEventListener('submit', startJobs);
	byId('job-form').addEventListener('change', (event) => {
		if (event.target.name === 'query-mode') updateQueryMode();
		if (event.target.name === 'pipeline') describeDepths();
	});
	byId('queue').addEventListener('click', (event) => {
		const runId = event.target.closest('button')?.dataset.dequeue;
		if (runId) dequeue(runId);
	});
	byId('rebuild').addEventListener('click', rebuild);
	byId('refresh-runs').addEventListener('click', loadRuns);
	byId('runs-filter').addEventListener('change', renderRuns);
	byId('runs-body').addEventListener('click', handleRunsClick);
	byId('runs-body').addEventListener('change', handleRunSelection);
	byId('score-selected').addEventListener('click', () => showScores([...selectedRuns]));
	byId('copy-json').addEventListener('click', () =>
		copyText(JSON.stringify(lastScores.raw, null, 2), 'JSON')
	);
	byId('copy-markdown').addEventListener('click', () => copyText(lastScores.markdown, 'Markdown'));
	byId('open-settings').addEventListener('click', openSettings);
	byId('close-settings').addEventListener('click', () => byId('settings-dialog').close());
	byId('reset-settings').addEventListener('click', () => renderSettingsFields(options.defaults));
	byId('settings-form').addEventListener('submit', saveSettings);
	byId('settings-form').addEventListener('input', (event) => {
		updateLiveHints();
		if (event.target.name?.startsWith('chunk')) loadCorpusCopies();
	});
	byId('settings-form').addEventListener('click', (event) => {
		const dataset = event.target.closest('button')?.dataset.prepare;
		if (dataset) prepareCorpus(dataset);
	});
}

async function start() {
	bindEvents();
	updateQueryMode();
	connectLog();
	options = await api('/api/options');
	renderOptions();
	await Promise.all([loadBuild(), refreshState()]);
	setInterval(refreshState, 2000);
}

start();
