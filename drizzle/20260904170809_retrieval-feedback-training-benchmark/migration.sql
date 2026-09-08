CREATE TABLE `retrieval_benchmark_cases` (
	`id` text PRIMARY KEY,
	`name` text(255) NOT NULL,
	`query` text NOT NULL,
	`document_ids` text NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `retrieval_benchmark_judgments` (
	`case_id` text NOT NULL,
	`chunk_id` text NOT NULL,
	`relevance` integer NOT NULL,
	`created_at` text NOT NULL,
	CONSTRAINT `retrieval_benchmark_judgments_pk` PRIMARY KEY(`case_id`, `chunk_id`),
	CONSTRAINT `fk_retrieval_benchmark_judgments_case_id_retrieval_benchmark_cases_id_fk` FOREIGN KEY (`case_id`) REFERENCES `retrieval_benchmark_cases`(`id`) ON DELETE CASCADE,
	CONSTRAINT `fk_retrieval_benchmark_judgments_chunk_id_document_chunks_id_fk` FOREIGN KEY (`chunk_id`) REFERENCES `document_chunks`(`id`) ON DELETE CASCADE,
	CONSTRAINT "retrieval_benchmark_judgments_relevance_check" CHECK("relevance" between 1 and 5)
);
--> statement-breakpoint
CREATE TABLE `retrieval_feedback` (
	`id` text PRIMARY KEY,
	`chunk_id` text NOT NULL,
	`impression_result_id` text,
	`query` text NOT NULL,
	`query_hash` text(64) NOT NULL,
	`rating` integer NOT NULL,
	`feedback_source` text DEFAULT 'human_expert' NOT NULL,
	`confidence` real,
	`rationale` text,
	`retrieval_mode` text NOT NULL,
	`result_rank` integer NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	CONSTRAINT `fk_retrieval_feedback_chunk_id_document_chunks_id_fk` FOREIGN KEY (`chunk_id`) REFERENCES `document_chunks`(`id`) ON DELETE CASCADE,
	CONSTRAINT `fk_retrieval_feedback_impression_result_id_retrieval_impression_results_id_fk` FOREIGN KEY (`impression_result_id`) REFERENCES `retrieval_impression_results`(`id`) ON DELETE SET NULL,
	CONSTRAINT "retrieval_feedback_rating_check" CHECK("rating" between 1 and 5),
	CONSTRAINT "retrieval_feedback_confidence_check" CHECK("confidence" is null or "confidence" between 0 and 1)
);
--> statement-breakpoint
CREATE TABLE `retrieval_impression_results` (
	`id` text PRIMARY KEY,
	`impression_id` text NOT NULL,
	`chunk_id` text NOT NULL,
	`retrieval_mode` text NOT NULL,
	`base_rank` integer NOT NULL,
	`displayed_rank` integer NOT NULL,
	`semantic_score` real,
	`bm25_score` real,
	`cross_encoder_score` real,
	`base_score` real NOT NULL,
	`learned_score` real,
	CONSTRAINT `fk_retrieval_impression_results_impression_id_retrieval_impressions_id_fk` FOREIGN KEY (`impression_id`) REFERENCES `retrieval_impressions`(`id`) ON DELETE CASCADE,
	CONSTRAINT `fk_retrieval_impression_results_chunk_id_document_chunks_id_fk` FOREIGN KEY (`chunk_id`) REFERENCES `document_chunks`(`id`) ON DELETE CASCADE
);
--> statement-breakpoint
CREATE TABLE `retrieval_impressions` (
	`id` text PRIMARY KEY,
	`query` text NOT NULL,
	`query_hash` text(64) NOT NULL,
	`requested_top_k` integer NOT NULL,
	`document_ids` text NOT NULL,
	`embedding_model` text NOT NULL,
	`reranker_model` text NOT NULL,
	`scoring_version` text NOT NULL,
	`ranker_model_id` text,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `retrieval_ranker_models` (
	`id` text PRIMARY KEY,
	`training_run_id` text NOT NULL,
	`feature_version` integer NOT NULL,
	`feature_names` text NOT NULL,
	`means` text NOT NULL,
	`standard_deviations` text NOT NULL,
	`weights` text NOT NULL,
	`intercept` real NOT NULL,
	`regularization` real NOT NULL,
	`epochs` integer NOT NULL,
	`training_loss` real NOT NULL,
	`created_at` text NOT NULL,
	CONSTRAINT `fk_retrieval_ranker_models_training_run_id_retrieval_training_runs_id_fk` FOREIGN KEY (`training_run_id`) REFERENCES `retrieval_training_runs`(`id`) ON DELETE CASCADE
);
--> statement-breakpoint
CREATE TABLE `retrieval_training_runs` (
	`id` text PRIMARY KEY,
	`status` text NOT NULL,
	`feedback_source` text DEFAULT 'human_expert' NOT NULL,
	`dataset_version` integer NOT NULL,
	`feature_version` integer NOT NULL,
	`embedding_model` text NOT NULL,
	`reranker_model` text NOT NULL,
	`scoring_version` text NOT NULL,
	`training_examples` integer NOT NULL,
	`validation_examples` integer NOT NULL,
	`distinct_queries` integer NOT NULL,
	`total_feedback` integer NOT NULL,
	`attributed_feedback` integer NOT NULL,
	`unattributed_feedback` integer NOT NULL,
	`inconsistent_feedback` integer NOT NULL,
	`hyperparameters` text NOT NULL,
	`evaluation` text,
	`error` text,
	`started_at` text NOT NULL,
	`completed_at` text
);
--> statement-breakpoint
ALTER TABLE `app_state` ADD `active_retrieval_model_id` text;--> statement-breakpoint
CREATE INDEX `retrieval_benchmark_cases_created_idx` ON `retrieval_benchmark_cases` (`created_at`);--> statement-breakpoint
CREATE INDEX `retrieval_benchmark_judgments_chunk_idx` ON `retrieval_benchmark_judgments` (`chunk_id`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_chunk_idx` ON `retrieval_feedback` (`chunk_id`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_query_idx` ON `retrieval_feedback` (`query_hash`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_source_idx` ON `retrieval_feedback` (`feedback_source`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_impression_result_idx` ON `retrieval_feedback` (`impression_result_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_feedback_chunk_query_source_idx` ON `retrieval_feedback` (`chunk_id`,`query_hash`,`feedback_source`);--> statement-breakpoint
CREATE INDEX `retrieval_impression_results_impression_idx` ON `retrieval_impression_results` (`impression_id`);--> statement-breakpoint
CREATE INDEX `retrieval_impression_results_chunk_idx` ON `retrieval_impression_results` (`chunk_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_impression_results_candidate_idx` ON `retrieval_impression_results` (`impression_id`,`retrieval_mode`,`chunk_id`);--> statement-breakpoint
CREATE INDEX `retrieval_impressions_query_idx` ON `retrieval_impressions` (`query_hash`);--> statement-breakpoint
CREATE INDEX `retrieval_impressions_created_idx` ON `retrieval_impressions` (`created_at`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_ranker_models_training_run_idx` ON `retrieval_ranker_models` (`training_run_id`);--> statement-breakpoint
CREATE INDEX `retrieval_ranker_models_created_idx` ON `retrieval_ranker_models` (`created_at`);--> statement-breakpoint
CREATE INDEX `retrieval_training_runs_status_idx` ON `retrieval_training_runs` (`status`);--> statement-breakpoint
CREATE INDEX `retrieval_training_runs_source_idx` ON `retrieval_training_runs` (`feedback_source`);--> statement-breakpoint
CREATE INDEX `retrieval_training_runs_started_idx` ON `retrieval_training_runs` (`started_at`);