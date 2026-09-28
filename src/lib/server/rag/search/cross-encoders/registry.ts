import type { CrossEncoder } from './cross-encoder';
import { Ettin } from './ettin';
import { MsMarco } from './ms-marco';

const builtInCrossEncoders: CrossEncoder[] = [new Ettin(), new MsMarco()];

export function listBuiltInCrossEncoders(): CrossEncoder[] {
	return [...builtInCrossEncoders];
}

export function findCrossEncoder(id: string): CrossEncoder | null {
	return builtInCrossEncoders.find((crossEncoder) => crossEncoder.id === id) ?? null;
}

export function requireCrossEncoder(id: string): CrossEncoder {
	const crossEncoder = findCrossEncoder(id);

	if (!crossEncoder) {
		throw new Error(`Cross-encoder "${id}" is unavailable.`);
	}

	return crossEncoder;
}
