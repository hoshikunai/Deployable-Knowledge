export const LAYOUT_LABELS = [
	'caption',
	'footnote',
	'formula',
	'list_item',
	'page_footer',
	'page_header',
	'picture',
	'section_header',
	'table',
	'text',
	'title',
	'document_index',
	'code',
	'checkbox_selected',
	'checkbox_unselected',
	'form',
	'key_value_region'
] as const;

export type LayoutLabel = (typeof LAYOUT_LABELS)[number];

export type Box = { l: number; t: number; r: number; b: number };

export type PageSize = { width: number; height: number };

export type LayoutDetection = { label: LayoutLabel; score: number; box: Box };

export type TextItemBox = {
	index: number;
	text: string;
	box: Box;
	hasEOL: boolean;
	fontHeight: number;
};

export type LayoutCluster = {
	id: number;
	label: LayoutLabel;
	confidence: number;
	box: Box;
	items: TextItemBox[];
	children: LayoutCluster[];
};

export type LayoutInput = { pixels: Uint8Array<ArrayBuffer>; width: number; height: number };

export type OutlineEntry = {
	level: number;
	label: string;
	pageIndex: number;
	x: number | null;
	y: number | null;
};

export type SectionAnchor = { level: number; label: string; x: number; y: number };
