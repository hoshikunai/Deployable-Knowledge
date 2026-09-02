import {
	getActiveRetrievalModelId,
	setActiveRetrievalModelId
} from '$lib/server/database/app-state';
import { EMBEDDING_MODEL } from '$lib/server/rag/embedding-model';
import { RETRIEVAL_FEATURE_NAMES } from '$lib/server/rag/training/build-retrieval-training-features';
import {
	LEARNED_RANKING_BLEND_WEIGHT,
	MAXIMUM_ACTIVATION_GROUP_NDCG_REGRESSION,
	MINIMUM_ACTIVATION_NDCG_IMPROVEMENT,
	MINIMUM_ACTIVATION_NON_TIE_WIN_RATE,
	MINIMUM_ACTIVATION_RANKING_GROUPS,
	RETRIEVAL_FEATURE_VERSION,
	RETRIEVAL_RANKING_STRATEGY
} from '$lib/server/rag/training/retrieval-training-constants';
import { RetrievalModelsRepository } from '$lib/server/repositories/retrieval-models.repository';
import { CROSS_ENCODER_MODEL, RETRIEVAL_SCORING_VERSION } from './retrieval-version';

export interface ActiveRetrievalModel {
	id: string;
	means: number[];
	standardDeviations: number[];
	weights: number[];
	intercept: number;
}

async function requireCompatibleRetrievalModel(modelId: string): Promise<ActiveRetrievalModel> {
	const stored = await RetrievalModelsRepository.findModelWithRunById(modelId);

	if (!stored) {
		throw new Error(`Retrieval model ${modelId} was not found.`);
	}

	const { model, run } = stored;
	const evaluation = run.evaluation;
	const expectedFeatureCount = RETRIEVAL_FEATURE_NAMES.length;

	if (run.status !== 'completed') {
		throw new Error(`Retrieval model ${modelId} does not have a completed training run.`);
	}

	if (
		run.embeddingModel !== EMBEDDING_MODEL ||
		run.rerankerModel !== CROSS_ENCODER_MODEL ||
		run.scoringVersion !== RETRIEVAL_SCORING_VERSION
	) {
		throw new Error(`Retrieval model ${modelId} is incompatible with the current search stack.`);
	}

	if (
		run.featureVersion !== RETRIEVAL_FEATURE_VERSION ||
		model.featureVersion !== RETRIEVAL_FEATURE_VERSION
	) {
		throw new Error(`Retrieval model ${modelId} uses an incompatible feature version.`);
	}

	if (
		model.featureNames.length !== expectedFeatureCount ||
		!model.featureNames.every(
			(featureName, index) => featureName === RETRIEVAL_FEATURE_NAMES[index]
		)
	) {
		throw new Error(`Retrieval model ${modelId} uses incompatible features.`);
	}

	if (
		model.means.length !== expectedFeatureCount ||
		model.standardDeviations.length !== expectedFeatureCount ||
		model.weights.length !== expectedFeatureCount
	) {
		throw new Error(`Retrieval model ${modelId} has invalid parameter dimensions.`);
	}

	if (
		!model.means.every(Number.isFinite) ||
		!model.standardDeviations.every((value) => Number.isFinite(value) && value > 0) ||
		!model.weights.every(Number.isFinite) ||
		!Number.isFinite(model.intercept)
	) {
		throw new Error(`Retrieval model ${modelId} contains invalid parameters.`);
	}

	if (
		!evaluation ||
		evaluation.rankingStrategy !== RETRIEVAL_RANKING_STRATEGY ||
		evaluation.blendWeight !== LEARNED_RANKING_BLEND_WEIGHT ||
		evaluation.ndcgImprovement === null ||
		evaluation.ndcgImprovement < MINIMUM_ACTIVATION_NDCG_IMPROVEMENT ||
		evaluation.evaluatedRankingGroups < MINIMUM_ACTIVATION_RANKING_GROUPS
	) {
		throw new Error(
			`Retrieval model ${modelId} does not meet the activation evaluation threshold.`
		);
	}

	const evaluatedNonTieGroups = evaluation.improvedRankingGroups + evaluation.degradedRankingGroups;
	const nonTieWinRate =
		evaluatedNonTieGroups > 0 ? evaluation.improvedRankingGroups / evaluatedNonTieGroups : 0;
	if (
		!Number.isInteger(evaluation.improvedRankingGroups) ||
		!Number.isInteger(evaluation.degradedRankingGroups) ||
		!Number.isInteger(evaluation.tiedRankingGroups) ||
		evaluation.improvedRankingGroups < 0 ||
		evaluation.degradedRankingGroups < 0 ||
		evaluation.tiedRankingGroups < 0 ||
		evaluation.improvedRankingGroups +
			evaluation.degradedRankingGroups +
			evaluation.tiedRankingGroups !==
			evaluation.evaluatedRankingGroups ||
		nonTieWinRate < MINIMUM_ACTIVATION_NON_TIE_WIN_RATE ||
		evaluation.worstRankingGroupNdcgDelta === null ||
		evaluation.worstRankingGroupNdcgDelta < -MAXIMUM_ACTIVATION_GROUP_NDCG_REGRESSION
	) {
		throw new Error(
			`Retrieval model ${modelId} does not generalize consistently enough for activation.`
		);
	}

	return {
		id: model.id,
		means: [...model.means],
		standardDeviations: [...model.standardDeviations],
		weights: [...model.weights],
		intercept: model.intercept
	};
}

export async function activateRetrievalModel(modelId: string): Promise<ActiveRetrievalModel> {
	const model = await requireCompatibleRetrievalModel(modelId);
	await setActiveRetrievalModelId(model.id);
	return model;
}

export async function deactivateRetrievalModel(): Promise<void> {
	await setActiveRetrievalModelId(null);
}

export async function loadActiveRetrievalModel(): Promise<ActiveRetrievalModel | null> {
	const modelId = await getActiveRetrievalModelId();
	if (!modelId) return null;

	try {
		return await requireCompatibleRetrievalModel(modelId);
	} catch (error) {
		console.warn('[Retrieval] Active learned ranker is unavailable.', error);
		return null;
	}
}
