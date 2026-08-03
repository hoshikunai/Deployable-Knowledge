<script lang="ts">
	import { RetrievalMode } from '$lib/enums';
	import { Button } from '$lib/components/ui/button';
	import * as ButtonGroup from '$lib/components/ui/button-group';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { settingsStore } from '$lib/stores';

	const modes = [
		{ value: RetrievalMode.SEMANTIC, label: 'Semantic' },
		{ value: RetrievalMode.BM25, label: 'BM25' },
		{ value: RetrievalMode.HYBRID, label: 'Hybrid' },
		{ value: RetrievalMode.KNOWLEDGE_GRAPH, label: 'Knowledge Graph' }
	];

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
				variant={settingsStore.config.retrievalMode === mode.value ? 'default' : 'outline'}
				size="sm"
				aria-pressed={settingsStore.config.retrievalMode === mode.value}
				onclick={() => settingsStore.updateConfig({ retrievalMode: mode.value })}
			>
				{mode.label}
			</Button>
		{/each}
	</ButtonGroup.Root>
	<div class="grid items-start gap-3 sm:grid-cols-2">
		<div class="grid min-w-0 grid-rows-[1.25rem_2.25rem] content-start gap-2">
			<Label for="settings-rag-top-k">Retrieved chunks</Label>
			<Input
				id="settings-rag-top-k"
				class="w-full"
				min="1"
				oninput={updateRagTopK}
				type="number"
				value={settingsStore.config.ragTopK}
			/>
		</div>
	</div>
</section>
