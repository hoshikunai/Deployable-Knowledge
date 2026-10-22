<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { renderMarkdown } from '$lib/utils/markdown';

	interface Props {
		class?: string;
		content: string;
		styledSpans?: boolean;
	}

	let { class: className, content, styledSpans = false }: Props = $props();
</script>

<div class={cn('markdown-content prose prose-sm max-w-none dark:prose-invert', className)}>
	{@html renderMarkdown(content, { styledSpans })}
</div>
