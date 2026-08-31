import { createHash } from 'node:crypto';
import { normalizeRetrievalQuery } from '$lib/utils';

export function hashRetrievalQuery(query: string): string {
	return createHash('sha256').update(normalizeRetrievalQuery(query)).digest('hex');
}
