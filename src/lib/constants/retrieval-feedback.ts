// Keep the legacy value readable so old records retain their provenance.
export const RETRIEVAL_FEEDBACK_SOURCES = ['human_expert', 'ai_proxy'] as const;

export type RetrievalFeedbackSource = (typeof RETRIEVAL_FEEDBACK_SOURCES)[number];

export const HUMAN_EXPERT_FEEDBACK_SOURCE = 'human_expert' satisfies RetrievalFeedbackSource;
