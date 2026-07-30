<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { RetrievalMode } from '$lib/enums';
	import { Button } from '$lib/components/ui/button';
	import * as ButtonGroup from '$lib/components/ui/button-group';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Progress } from '$lib/components/ui/progress';
	import { hippoRagStore, settingsStore } from '$lib/stores';

	const modes = [
		{ value: RetrievalMode.SEMANTIC, label: 'Semantic' },
		{ value: RetrievalMode.BM25, label: 'BM25' },
		{ value: RetrievalMode.HYBRID, label: 'Hybrid' },
		{ value: RetrievalMode.HIPPORAG_2, label: 'HippoRAG 2' }
	];

	const indexDescription = $derived.by(() => {
		const status = hippoRagStore.status;
		if (!status) return 'Index status unavailable.';
		if (status.totalChunks === 0) return 'Add documents before building the graph index.';
		if (status.ready) return `${status.indexedChunks} chunks indexed and ready.`;
		if (status.indexedChunks === 0)
			return `${status.totalChunks} chunks are waiting to be indexed.`;
		return `${status.indexedChunks} indexed; ${status.pendingChunks} waiting.`;
	});

	onMount(() => {
		void hippoRagStore.load();
	});

	async function buildIndex(rebuild: boolean): Promise<void> {
		try {
			await settingsStore.saveActive();
			await hippoRagStore.build(rebuild);
			toast.success('HippoRAG2 index is ready');
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'HippoRAG2 index build failed');
		}
	}

	function updateRagTopK(event: Event): void {
		if (!(event.currentTarget instanceof HTMLInputElement)) return;
		const value = event.currentTarget.valueAsNumber;
		if (Number.isFinite(value)) {
			settingsStore.updateConfig({ ragTopK: Math.max(1, Math.floor(value)) });
		}
	}
</script>

<section class="grid gap-3">
	<Label>Retrieval</Label>
	<ButtonGroup.Root class="w-fit">
		{#each modes as mode (mode.value)}
			<Button
				aria-pressed={settingsStore.config.retrievalMode === mode.value}
				onclick={() => settingsStore.updateConfig({ retrievalMode: mode.value })}
				size="sm"
				variant={settingsStore.config.retrievalMode === mode.value ? 'default' : 'outline'}
			>
				{mode.label}
			</Button>
		{/each}
	</ButtonGroup.Root>
	<div class="grid max-w-52 gap-2">
		<Label for="settings-rag-top-k">Retrieved chunks</Label>
		<Input
			id="settings-rag-top-k"
			min="1"
			oninput={updateRagTopK}
			type="number"
			value={settingsStore.config.ragTopK}
		/>
	</div>
	{#if settingsStore.config.retrievalMode === RetrievalMode.HIPPORAG_2}
		<div class="dk-panel grid max-w-2xl gap-3 rounded-lg border p-3">
			<div class="grid gap-1">
				<p class="text-sm font-medium">HippoRAG2 graph index</p>
				<p class="text-xs text-muted-foreground">
					{indexDescription} Indexing uses the selected provider and model for OpenIE; queries use the
					same model for fact recognition. The graph and ranking pipeline run in TypeScript without Python.
					Dense retrieval is used automatically if the graph is unavailable.
				</p>
				{#if hippoRagStore.status?.providerId && hippoRagStore.status.modelId}
					<p class="text-xs text-muted-foreground">
						Built with {hippoRagStore.status.providerId} / {hippoRagStore.status.modelId}
					</p>
				{/if}
			</div>
			{#if hippoRagStore.progress}
				<div aria-live="polite" class="grid gap-1.5">
					<Progress value={hippoRagStore.progress.percent} />
					<p class="text-xs text-muted-foreground">{hippoRagStore.progress.message}</p>
				</div>
			{/if}
			{#if hippoRagStore.error}
				<p class="text-xs text-destructive" role="alert">{hippoRagStore.error}</p>
			{/if}
			<div class="flex flex-wrap gap-2">
				<Button
					disabled={hippoRagStore.building || !hippoRagStore.status?.totalChunks}
					onclick={() => void buildIndex(false)}
					size="sm"
				>
					{hippoRagStore.status?.indexedChunks ? 'Update index' : 'Build index'}
				</Button>
				<Button
					disabled={hippoRagStore.building || !hippoRagStore.status?.totalChunks}
					onclick={() => void buildIndex(true)}
					size="sm"
					variant="outline"
				>
					Rebuild all
				</Button>
			</div>
		</div>
	{/if}
</section>
