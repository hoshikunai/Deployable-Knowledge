import type { SearchMatchBase } from '../search/search-shared';

export type HippoTriple = {
	subject: string;
	predicate: string;
	object: string;
};

export type HippoOpenIeResult = {
	entities: string[];
	triples: HippoTriple[];
};

export type HippoIndexConfig = {
	providerId: string;
	modelId: string;
};

export type HippoIndexProgress = {
	stage: 'extracting' | 'embedding' | 'linking' | 'finalizing';
	current: number;
	total: number;
	message: string;
};

export type HippoPassageCandidate = SearchMatchBase & {
	embedding: Uint8Array | ArrayBuffer;
};
