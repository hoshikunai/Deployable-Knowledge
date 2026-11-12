<script lang="ts">
	import FileScan from '@lucide/svelte/icons/file-scan';
	import { onMount } from 'svelte';
	import { Checkbox } from '#lib/components/ui/checkbox';
	import { Label } from '#lib/components/ui/label';
	import { Progress } from '#lib/components/ui/progress';
	import { Skeleton } from '#lib/components/ui/skeleton';
	import { documentParsingStore } from '#lib/stores';
	import { formatBytes } from '#lib/utils';
	import SettingsFieldGroup from './SettingsFieldGroup.svelte';

	const status = $derived(documentParsingStore.status);
	const busy = $derived(documentParsingStore.saving || documentParsingStore.downloading);
	const modelNote = $derived.by(() => {
		if (!status) return '';
		if (status.modelInstalled) return 'The layout model is downloaded.';
		return `Turning this on downloads the layout model (${formatBytes(status.modelSizeBytes)}).`;
	});

	onMount(() => void documentParsingStore.refresh());
</script>

{#if status}
	<SettingsFieldGroup
		icon={FileScan}
		title="PDF parsing"
		hint="Applies to PDFs, Word documents and presentations added from now on."
	>
		<div class="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-2.5">
			<Checkbox
				checked={status.structuredPdfParsing}
				disabled={busy}
				id="settings-structured-pdf-parsing"
				onCheckedChange={(checked) =>
					void documentParsingStore.setStructuredPdfParsing(checked === true)}
			/>
			<div class="grid gap-0.5">
				<Label class="text-sm font-medium" for="settings-structured-pdf-parsing">
					Structure-aware parsing
				</Label>
				<p class="m-0 text-xs text-muted-foreground">
					Detects headings, tables, and running headers on each page so chunks follow the document's
					sections. Ingesting takes roughly half a second longer per page. {modelNote}
				</p>
			</div>
		</div>
		{#if documentParsingStore.progress}
			<div class="grid gap-1.5">
				<Progress value={documentParsingStore.progress.percent} />
				<p class="m-0 text-xs text-muted-foreground">
					{documentParsingStore.progress.label}: {documentParsingStore.progress.message}
				</p>
			</div>
		{/if}
		{#if documentParsingStore.error}
			<p class="m-0 text-sm text-destructive" role="alert">{documentParsingStore.error}</p>
		{/if}
	</SettingsFieldGroup>
{:else}
	<Skeleton class="h-24 w-full" />
{/if}
