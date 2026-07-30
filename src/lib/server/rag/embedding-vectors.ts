export function embeddingToBuffer(values: readonly number[]): Buffer {
	const array = Float32Array.from(values);
	return Buffer.from(array.buffer, array.byteOffset, array.byteLength);
}

export function embeddingFromBytes(value: Uint8Array | ArrayBuffer): Float32Array {
	const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);

	if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
		throw new Error('Embedding bytes are not aligned to Float32 values.');
	}

	return new Float32Array(
		bytes.buffer,
		bytes.byteOffset,
		bytes.byteLength / Float32Array.BYTES_PER_ELEMENT
	);
}

export function embeddingDotProduct(left: ArrayLike<number>, right: ArrayLike<number>): number {
	if (left.length !== right.length) {
		throw new Error(`Embedding dimensions do not match (${left.length} and ${right.length}).`);
	}

	let score = 0;
	for (let index = 0; index < left.length; index += 1) {
		score += left[index] * right[index];
	}
	return score;
}
