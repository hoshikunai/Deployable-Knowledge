<script lang="ts">
	import Table from '@lucide/svelte/icons/table';
	import { onMount } from 'svelte';
	import { Checkbox } from '#lib/components/ui/checkbox';
	import { Label } from '#lib/components/ui/label';
	import { Progress } from '#lib/components/ui/progress';
	import { Skeleton } from '#lib/components/ui/skeleton';
	import { tableRecognitionStore } from '#lib/stores';
	import { formatBytes } from '#lib/utils';
	import SettingsFieldGroup from './SettingsFieldGroup.svelte';

	const status = $derived(tableRecognitionStore.status);
	const busy = $derived(tableRecognitionStore.saving || tableRecognitionStore.downloading);
	const modelNote = $derived.by(() => {
		if (!status) return '';
		if (status.modelsInstalled) return 'The table models are downloaded.';
		return `Turning this on downloads the table models (${formatBytes(status.modelsSizeBytes)}).`;
	});

	onMount(() => void tableRecognitionStore.refresh());
</script>

{#if status}
	<SettingsFieldGroup
		icon={Table}
		title="PDF tables"
		hint="Applies to PDFs, Word documents and presentations added from now on."
	>
		<div class="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-2.5">
			<Checkbox
				checked={status.enabled}
				disabled={busy}
				id="settings-table-recognition"
				onCheckedChange={(checked) => void tableRecognitionStore.setEnabled(checked === true)}
			/>
			<div class="grid gap-0.5">
				<Label class="text-sm font-medium" for="settings-table-recognition">
					Table recognition
				</Label>
				<p class="m-0 text-xs text-muted-foreground">
					Finds tables on each page and reads their rows, columns and headers, so search can match a
					value to its row and column. Ingesting takes about half a second longer per page, plus
					about half a second per table. {modelNote}
				</p>
			</div>
		</div>
		{#if tableRecognitionStore.progress}
			<div class="grid gap-1.5">
				<Progress value={tableRecognitionStore.progress.percent} />
				<p class="m-0 text-xs text-muted-foreground">
					{tableRecognitionStore.progress.label}: {tableRecognitionStore.progress.message}
				</p>
			</div>
		{/if}
		{#if tableRecognitionStore.error}
			<p class="m-0 text-sm text-destructive" role="alert">{tableRecognitionStore.error}</p>
		{/if}
	</SettingsFieldGroup>
{:else}
	<Skeleton class="h-24 w-full" />
{/if}
