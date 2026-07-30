import { error, json } from '@sveltejs/kit';
import type { ApiHippoRagBuildEvent, ApiHippoRagBuildRequest } from '$lib/types';
import { seedLocalUser } from '$lib/server/database/seed';
import { ProfilesRepository } from '$lib/server/repositories';
import { buildHippoIndex, getHippoIndexStatus } from '$lib/server/rag/hipporag';
import type { HippoIndexProgress } from '$lib/server/rag/hipporag/types';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => json(await getHippoIndexStatus());

function progressPercent(progress: HippoIndexProgress): number {
	const ratio = progress.total > 0 ? progress.current / progress.total : 1;
	if (progress.stage === 'linking') return 80 + ratio * 19;
	if (progress.stage === 'finalizing') return 100;
	return ratio * 80;
}

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json().catch(() => ({}))) as ApiHippoRagBuildRequest;
	const user = await seedLocalUser();
	const profile = await ProfilesRepository.getActive(user);
	if (!profile?.provider.trim() || !profile.model.trim()) {
		throw error(400, 'Select and save an AI provider and model before building the index.');
	}

	const abortController = new AbortController();
	let closed = false;
	const stream = new ReadableStream({
		start(controller) {
			const encoder = new TextEncoder();
			const send = (event: ApiHippoRagBuildEvent) => {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
				} catch {
					closed = true;
				}
			};

			void (async () => {
				try {
					const result = await buildHippoIndex(
						{ providerId: profile.provider, modelId: profile.model },
						body.rebuild === true,
						(progress) =>
							send({
								status: 'progress',
								percent: progressPercent(progress),
								label: 'Building HippoRAG2 index',
								message: progress.message
							}),
						abortController.signal
					);
					send({ status: 'complete', result });
				} catch (cause) {
					console.error('HippoRAG2 index build failed', cause);
					send({
						status: 'error',
						message: cause instanceof Error ? cause.message : 'HippoRAG2 index build failed'
					});
				} finally {
					if (!closed) controller.close();
				}
			})().catch((cause) => console.error('HippoRAG2 index stream failed', cause));
		},
		cancel() {
			closed = true;
			abortController.abort();
		}
	});

	return new Response(stream, {
		headers: {
			'Cache-Control': 'no-cache',
			'Content-Type': 'application/x-ndjson; charset=utf-8',
			'X-Accel-Buffering': 'no'
		}
	});
};
