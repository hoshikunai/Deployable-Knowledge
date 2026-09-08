ALTER TABLE `retrieval_impression_results` ADD `was_displayed` integer DEFAULT true NOT NULL;--> statement-breakpoint
PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_retrieval_impression_results` (
	`id` text PRIMARY KEY,
	`impression_id` text NOT NULL,
	`chunk_id` text NOT NULL,
	`retrieval_mode` text NOT NULL,
	`base_rank` integer NOT NULL,
	`displayed_rank` integer,
	`was_displayed` integer DEFAULT true NOT NULL,
	`semantic_score` real,
	`bm25_score` real,
	`cross_encoder_score` real,
	`base_score` real NOT NULL,
	`learned_score` real,
	CONSTRAINT `fk_retrieval_impression_results_impression_id_retrieval_impressions_id_fk` FOREIGN KEY (`impression_id`) REFERENCES `retrieval_impressions`(`id`) ON DELETE CASCADE,
	CONSTRAINT `fk_retrieval_impression_results_chunk_id_document_chunks_id_fk` FOREIGN KEY (`chunk_id`) REFERENCES `document_chunks`(`id`) ON DELETE CASCADE,
	CONSTRAINT "retrieval_impression_results_display_state_check" CHECK("was_displayed" = ("displayed_rank" is not null))
);
--> statement-breakpoint
INSERT INTO `__new_retrieval_impression_results`(`id`, `impression_id`, `chunk_id`, `retrieval_mode`, `base_rank`, `displayed_rank`, `semantic_score`, `bm25_score`, `cross_encoder_score`, `base_score`, `learned_score`) SELECT `id`, `impression_id`, `chunk_id`, `retrieval_mode`, `base_rank`, `displayed_rank`, `semantic_score`, `bm25_score`, `cross_encoder_score`, `base_score`, `learned_score` FROM `retrieval_impression_results`;--> statement-breakpoint
DROP TABLE `retrieval_impression_results`;--> statement-breakpoint
ALTER TABLE `__new_retrieval_impression_results` RENAME TO `retrieval_impression_results`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
CREATE INDEX `retrieval_impression_results_impression_idx` ON `retrieval_impression_results` (`impression_id`);--> statement-breakpoint
CREATE INDEX `retrieval_impression_results_chunk_idx` ON `retrieval_impression_results` (`chunk_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `retrieval_impression_results_candidate_idx` ON `retrieval_impression_results` (`impression_id`,`retrieval_mode`,`chunk_id`);