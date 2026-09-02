export type RetrievalTrainingRunStatus = 'training' | 'completed' | 'failed';

export interface RetrievalTrainingHyperparameters {
	learningRate: number;
	l2Regularization: number;
	maxEpochs: number;
	earlyStoppingPatience: number;
	lossTolerance: number;
	validationFraction: number;
	ratingBalanceStrategy: string;
}

export interface RetrievalTrainingEvaluation {
	rankingStrategy: string;
	blendWeight: number;
	meanAbsoluteError: number;
	baselineNdcgAt5: number | null;
	trainedNdcgAt5: number | null;
	ndcgImprovement: number | null;
	evaluatedExamples: number;
	evaluatedRankingGroups: number;
	improvedRankingGroups: number;
	degradedRankingGroups: number;
	tiedRankingGroups: number;
	worstRankingGroupNdcgDelta: number | null;
}
