export type RetrievalTrainingRunStatus = 'training' | 'completed' | 'failed';

export interface RetrievalTrainingHyperparameters {
	algorithm: string;
	learningRate: number;
	l2Regularization: number;
	maxEpochs: number;
	earlyStoppingPatience: number;
	lossTolerance: number;
	crossValidationFolds: number;
	pairWeightingStrategy: string;
}

export interface RetrievalTrainingEvaluation {
	rankingStrategy: string;
	blendWeight: number;
	crossValidationFolds: number;
	pairwiseAccuracy: number;
	evaluatedPairs: number;
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
