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
CREATE UNIQUE INDEX `retrieval_ranker_models_training_run_idx` ON `retrieval_ranker_models` (`training_run_id`);--> statement-breakpoint
CREATE INDEX `retrieval_ranker_models_created_idx` ON `retrieval_ranker_models` (`created_at`);--> statement-breakpoint
CREATE INDEX `retrieval_training_runs_status_idx` ON `retrieval_training_runs` (`status`);--> statement-breakpoint
CREATE INDEX `retrieval_training_runs_started_idx` ON `retrieval_training_runs` (`started_at`);