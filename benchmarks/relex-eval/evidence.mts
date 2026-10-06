import type { RelexEntity } from '../../src/lib/server/entities/relex';

const BOUNDARY = /[.!?\n•○]/;

export function evidence(text: string, head: RelexEntity, tail: RelexEntity): string {
	let start = Math.min(head.start, tail.start);
	let end = Math.max(head.end, tail.end);
	while (start > 0 && !BOUNDARY.test(text[start - 1])) start -= 1;
	while (end < text.length && !BOUNDARY.test(text[end])) end += 1;
	return text
		.slice(start, Math.min(text.length, end + 1))
		.replace(/\s+/g, ' ')
		.trim();
}
