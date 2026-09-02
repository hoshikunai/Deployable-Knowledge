ALTER TABLE `retrieval_feedback` ADD `feedback_source` text DEFAULT 'human_expert' NOT NULL;--> statement-breakpoint
ALTER TABLE `retrieval_feedback` ADD `confidence` real;--> statement-breakpoint
ALTER TABLE `retrieval_feedback` ADD `rationale` text;--> statement-breakpoint
ALTER TABLE `retrieval_training_runs` ADD `feedback_source` text DEFAULT 'human_expert' NOT NULL;--> statement-breakpoint
PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_retrieval_feedback` (
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
INSERT INTO `__new_retrieval_feedback`(`id`, `chunk_id`, `impression_result_id`, `query`, `query_hash`, `rating`, `retrieval_mode`, `result_rank`, `created_at`, `updated_at`) SELECT `id`, `chunk_id`, `impression_result_id`, `query`, `query_hash`, `rating`, `retrieval_mode`, `result_rank`, `created_at`, `updated_at` FROM `retrieval_feedback`;--> statement-breakpoint
DROP TABLE `retrieval_feedback`;--> statement-breakpoint
ALTER TABLE `__new_retrieval_feedback` RENAME TO `retrieval_feedback`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
DROP INDEX IF EXISTS `retrieval_feedback_chunk_query_idx`;--> statement-breakpoint
CREATE INDEX `retrieval_feedback_chunk_idx` ON `retrieval_feedback` (`chunk_id`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_query_idx` ON `retrieval_feedback` (`query_hash`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_source_idx` ON `retrieval_feedback` (`feedback_source`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_impression_result_idx` ON `retrieval_feedback` (`impression_result_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_feedback_chunk_query_source_idx` ON `retrieval_feedback` (`chunk_id`,`query_hash`,`feedback_source`);--> statement-breakpoint
CREATE INDEX `retrieval_training_runs_source_idx` ON `retrieval_training_runs` (`feedback_source`);