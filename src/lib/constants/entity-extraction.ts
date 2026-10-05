export interface RelexThresholds {
	entity: number;
	adjacency: number;
	relation: number;
}

export const DEFAULT_RELEX_THRESHOLDS: Readonly<RelexThresholds> = {
	entity: 0.5,
	adjacency: 0.55,
	relation: 0.7
};
