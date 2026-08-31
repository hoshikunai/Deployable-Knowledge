CREATE TABLE `retrieval_feedback` (
	`id` text PRIMARY KEY,
	`chunk_id` text NOT NULL,
	`query` text NOT NULL,
	`query_hash` text(64) NOT NULL,
	`rating` integer NOT NULL,
	`retrieval_mode` text NOT NULL,
	`result_rank` integer NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	CONSTRAINT `fk_retrieval_feedback_chunk_id_document_chunks_id_fk` FOREIGN KEY (`chunk_id`) REFERENCES `document_chunks`(`id`) ON DELETE CASCADE,
	CONSTRAINT "retrieval_feedback_rating_check" CHECK("rating" between 1 and 5)
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
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `retrieval_feedback_chunk_idx` ON `retrieval_feedback` (`chunk_id`);--> statement-breakpoint
CREATE INDEX `retrieval_feedback_query_idx` ON `retrieval_feedback` (`query_hash`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_feedback_chunk_query_idx` ON `retrieval_feedback` (`chunk_id`,`query_hash`);--> statement-breakpoint
CREATE INDEX `retrieval_impression_results_impression_idx` ON `retrieval_impression_results` (`impression_id`);--> statement-breakpoint
CREATE INDEX `retrieval_impression_results_chunk_idx` ON `retrieval_impression_results` (`chunk_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_impression_results_candidate_idx` ON `retrieval_impression_results` (`impression_id`,`retrieval_mode`,`chunk_id`);--> statement-breakpoint
CREATE INDEX `retrieval_impressions_query_idx` ON `retrieval_impressions` (`query_hash`);--> statement-breakpoint
CREATE INDEX `retrieval_impressions_created_idx` ON `retrieval_impressions` (`created_at`);