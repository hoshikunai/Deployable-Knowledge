import { browser } from '$app/environment';
import MarkdownIt from 'markdown-it';
import type StateInline from 'markdown-it/lib/rules_inline/state_inline.mjs';
import DOMPurify from 'dompurify';
import { matchStyledSpan, textStyleClasses } from '$lib/utils/notebook-styles';

function createEngine(): MarkdownIt {
	const engine = new MarkdownIt({ html: false, linkify: true });
	engine.renderer.rules.link_open = (tokens, idx, options, _env, self) => {
		tokens[idx].attrSet('target', '_blank');
		tokens[idx].attrSet('rel', 'noopener noreferrer');
		return self.renderToken(tokens, idx, options);
	};
	return engine;
}

// Renders the notebook's `[text]{color=… hl=…}` as a classed span,
// tokenizing the bracketed text as ordinary inline markdown
function styledSpanRule(state: StateInline, silent: boolean): boolean {
	const match = matchStyledSpan(state.src, state.pos);
	if (!match || match.end > state.posMax) return false;
	if (!silent) {
		const { posMax } = state;
		state.push('styled_span_open', 'span', 1).attrSet('class', textStyleClasses(match.style));
		state.pos = match.open + 1;
		state.posMax = match.close;
		state.md.inline.tokenize(state);
		state.posMax = posMax;
		state.push('styled_span_close', 'span', -1);
	}
	state.pos = match.end;
	return true;
}

const md = createEngine();
const notebookMd = createEngine();
notebookMd.inline.ruler.before('link', 'styled_span', styledSpanRule);

export interface RenderMarkdownOptions {
	styledSpans?: boolean;
}

export function renderMarkdown(text = '', options: RenderMarkdownOptions = {}): string {
	const html = (options.styledSpans ? notebookMd : md).render(text.trim());
	if (!browser) return html;
	return DOMPurify.sanitize(html);
}
