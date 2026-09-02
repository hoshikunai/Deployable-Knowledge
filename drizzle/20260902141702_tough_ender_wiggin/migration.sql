ALTER TABLE `app_state` ADD `active_retrieval_model_id` text;--> statement-breakpoint
ALTER TABLE `retrieval_impression_results` ADD `learned_score` real;--> statement-breakpoint
ALTER TABLE `retrieval_impressions` ADD `ranker_model_id` text;