<script lang="ts">
	import Bold from '@lucide/svelte/icons/bold';
	import Code from '@lucide/svelte/icons/code';
	import Heading1 from '@lucide/svelte/icons/heading-1';
	import Highlighter from '@lucide/svelte/icons/highlighter';
	import Heading2 from '@lucide/svelte/icons/heading-2';
	import Heading3 from '@lucide/svelte/icons/heading-3';
	import Italic from '@lucide/svelte/icons/italic';
	import Link from '@lucide/svelte/icons/link';
	import List from '@lucide/svelte/icons/list';
	import ListOrdered from '@lucide/svelte/icons/list-ordered';
	import Palette from '@lucide/svelte/icons/palette';
	import SquareCode from '@lucide/svelte/icons/square-code';
	import Strikethrough from '@lucide/svelte/icons/strikethrough';
	import TextQuote from '@lucide/svelte/icons/text-quote';
	import { ActionIcon } from '$lib/components/app/actions';
	import * as Popover from '$lib/components/ui/popover';
	import { Separator } from '$lib/components/ui/separator';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		NOTEBOOK_HIGHLIGHTS,
		NOTEBOOK_TEXT_COLORS,
		type NotebookHighlight,
		type NotebookTextColor
	} from '$lib/constants';
	import type { RibbonCommand } from './notebook-editing';

	interface Props {
		onCommand: (command: RibbonCommand) => void;
	}

	let { onCommand }: Props = $props();
	let colorOpen = $state(false);
	let highlightOpen = $state(false);

	const SWATCH_CLASS =
		'size-6 cursor-pointer rounded-full border border-border/60 transition-transform hover:scale-110 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none';

	function pickColor(color: NotebookTextColor): void {
		colorOpen = false;
		onCommand({ color });
	}

	function pickHighlight(highlight: NotebookHighlight): void {
		highlightOpen = false;
		onCommand({ highlight });
	}
</script>

<div
	class="flex flex-wrap items-center gap-0.5 border-b bg-card/50 px-2 py-1"
	role="toolbar"
	aria-label="Text formatting"
>
	<ActionIcon class="size-7" label="Bold (Ctrl+B)" variant="ghost" onclick={() => onCommand('bold')}
		><Bold /></ActionIcon
	>
	<ActionIcon
		class="size-7"
		label="Italic (Ctrl+I)"
		variant="ghost"
		onclick={() => onCommand('italic')}><Italic /></ActionIcon
	>
	<ActionIcon
		class="size-7"
		label="Strikethrough"
		variant="ghost"
		onclick={() => onCommand('strikethrough')}><Strikethrough /></ActionIcon
	>
	<Popover.Root bind:open={colorOpen}>
		<Popover.Trigger>
			{#snippet child({ props })}
				<ActionIcon class="size-7" label="Text color" triggerProps={props} variant="ghost"
					><Palette /></ActionIcon
				>
			{/snippet}
		</Popover.Trigger>
		<Popover.Content align="start" class="w-auto p-2">
			<div class="grid grid-cols-4 gap-1.5">
				{#each NOTEBOOK_TEXT_COLORS as color (color.name)}
					<Tooltip.Root>
						<Tooltip.Trigger
							type="button"
							class={SWATCH_CLASS}
							style={`background-color: var(--nb-color-${color.name})`}
							aria-label={`Color text ${color.label.toLowerCase()}`}
							onclick={() => pickColor(color.name)}
						></Tooltip.Trigger>
						<Tooltip.Content>{color.label}</Tooltip.Content>
					</Tooltip.Root>
				{/each}
			</div>
		</Popover.Content>
	</Popover.Root>
	<Popover.Root bind:open={highlightOpen}>
		<Popover.Trigger>
			{#snippet child({ props })}
				<ActionIcon class="size-7" label="Highlight" triggerProps={props} variant="ghost"
					><Highlighter /></ActionIcon
				>
			{/snippet}
		</Popover.Trigger>
		<Popover.Content align="start" class="w-auto p-2">
			<div class="grid grid-cols-4 gap-1.5">
				{#each NOTEBOOK_HIGHLIGHTS as highlight (highlight.name)}
					<Tooltip.Root>
						<Tooltip.Trigger
							type="button"
							class={SWATCH_CLASS}
							style={`background-color: var(--nb-highlight-${highlight.name})`}
							aria-label={`Highlight text ${highlight.label.toLowerCase()}`}
							onclick={() => pickHighlight(highlight.name)}
						></Tooltip.Trigger>
						<Tooltip.Content>{highlight.label}</Tooltip.Content>
					</Tooltip.Root>
				{/each}
			</div>
		</Popover.Content>
	</Popover.Root>
	<Separator orientation="vertical" class="mx-1 h-5!" />
	<ActionIcon class="size-7" label="Heading 1" variant="ghost" onclick={() => onCommand('h1')}
		><Heading1 /></ActionIcon
	>
	<ActionIcon class="size-7" label="Heading 2" variant="ghost" onclick={() => onCommand('h2')}
		><Heading2 /></ActionIcon
	>
	<ActionIcon class="size-7" label="Heading 3" variant="ghost" onclick={() => onCommand('h3')}
		><Heading3 /></ActionIcon
	>
	<Separator orientation="vertical" class="mx-1 h-5!" />
	<ActionIcon
		class="size-7"
		label="Bulleted list"
		variant="ghost"
		onclick={() => onCommand('bullet-list')}><List /></ActionIcon
	>
	<ActionIcon
		class="size-7"
		label="Numbered list"
		variant="ghost"
		onclick={() => onCommand('ordered-list')}><ListOrdered /></ActionIcon
	>
	<ActionIcon class="size-7" label="Quote" variant="ghost" onclick={() => onCommand('quote')}
		><TextQuote /></ActionIcon
	>
	<Separator orientation="vertical" class="mx-1 h-5!" />
	<ActionIcon class="size-7" label="Inline code" variant="ghost" onclick={() => onCommand('code')}
		><Code /></ActionIcon
	>
	<ActionIcon
		class="size-7"
		label="Code block"
		variant="ghost"
		onclick={() => onCommand('code-block')}><SquareCode /></ActionIcon
	>
	<ActionIcon class="size-7" label="Link" variant="ghost" onclick={() => onCommand('link')}
		><Link /></ActionIcon
	>
</div>
