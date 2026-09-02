export const RETRIEVAL_FEATURE_VERSION = 1 as const;

export const RETRIEVAL_TRAINING_ALGORITHM = 'pairwise-logistic-v1' as const;
export const RETRIEVAL_RANKING_STRATEGY = 'pairwise-logistic-rank-blend-v2' as const;
export const LEARNED_RANKING_BLEND_WEIGHT = 0.35;
export const RETRIEVAL_PAIR_WEIGHTING_STRATEGY = 'rating-gap-equal-impression' as const;
export const CROSS_VALIDATION_FOLD_COUNT = 5;

export const MINIMUM_TRAINING_EXAMPLES = 30;
export const MINIMUM_DISTINCT_QUERIES = 10;
export const MINIMUM_ACTIVATION_RANKING_GROUPS = 10;
export const MINIMUM_ACTIVATION_NDCG_IMPROVEMENT = 0.005;
export const MINIMUM_ACTIVATION_NON_TIE_WIN_RATE = 0.6;
export const MAXIMUM_ACTIVATION_GROUP_NDCG_REGRESSION = 0.02;

export const TRAINING_LEARNING_RATE = 0.01;
export const TRAINING_L2_REGULARIZATION = 0.1;
export const TRAINING_MAX_EPOCHS = 2_000;
export const TRAINING_EARLY_STOPPING_PATIENCE = 50;
export const TRAINING_LOSS_TOLERANCE = 1e-8;
