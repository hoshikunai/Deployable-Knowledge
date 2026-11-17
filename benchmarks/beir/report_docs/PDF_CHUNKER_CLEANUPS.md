# PDF chunker cleanups: what they change and what they measured

**Status:** reverted on `feature/docling-beir` on 2026-11-16. The PDF chunker now matches upstream
`main` (`Deployable-Knowledge/Deployable-Knowledge`, `d1f5ce11`). The code shown below is still in
commit `07891439` on `test/docling`, and in `HEAD` (`f09e557b`) of this branch.

**Decision needed:** whether to bring any of these cleanups back into the default PDF path. They
would change how every PDF a user ingests is chunked, whether table recognition is on or off.

## Summary

Four "Step 1" cleanups were added to the PDF chunker during the Docling work, then kept when the
layout pipeline was removed. They were never measured on their own until now.

| #   | Cleanup                         | Changes                                                              | Chunk types            |
| --- | ------------------------------- | -------------------------------------------------------------------- | ---------------------- |
| 1   | Running header and footer strip | Deletes recurring top and bottom lines of each page                  | TEXT                   |
| 2   | Cross-page sentence join        | Moves a sentence's tail back onto the page where the sentence starts | TEXT                   |
| 3   | Duplicate pdf-parse table drop  | Skips a pdf-parse table whose cells are all in the page text         | TABLE (pdf-parse only) |
| 4   | Long OCR split                  | Cuts image OCR text over 1,200 characters into sentence windows      | IMAGE                  |

Measured on `pdf-eval` with the app default pipeline (81 questions, page-level scoring):

- **Semantic search ranks the right page higher.** Semantic MRR@10 rises from 0.613 to 0.660
  (+0.047, +7.7%, **p = 0.027**), and nDCG@10 rises +0.022 (p = 0.12). Recall@10 falls slightly
  (−0.019).
- **With table recognition on as well, both semantic and hybrid search improve significantly
  over upstream.**
  - Semantic: nDCG@10 0.646 → 0.688 (**p = 0.040**), MRR@10 0.613 → 0.680 (**p = 0.010**).
  - Hybrid: nDCG@10 0.787 → 0.823 (**p = 0.045**), MRR@10 0.781 → 0.826 (p = 0.051).

  Of the four configurations, this is the only one that beats upstream significantly.

- **On their own, the cleanups don't improve hybrid search**, which is what the app uses.
  Hybrid nDCG@10 is flat (+0.003, p = 0.84), and Recall@10 drops from 0.905 to 0.875.

Only cleanups 1 and 2 had any effect on `pdf-eval`. Cleanup 3 had nothing to act on, because
pdf-parse finds no ruled tables in these PDFs. Cleanup 4 never triggered: the IMAGE chunk count is
43 in both copies.

## Code changes

The diffs are against upstream `main`. Lines with `+` are what the cleanups add.

### 1 and 2: new file `src/lib/server/rag/chunk/page-cleanup.ts`

Both functions only look at TEXT chunks. Each PDF page arrives as one TEXT chunk before
chunking, plus any IMAGE and TABLE chunks.

```ts
const EDGE_LINES = 2;
const MIN_BANNER_PAGES = 3;
const BANNER_PAGE_RATIO = 0.3;
const MIN_PAGES_FOR_BANNERS = 4;
const MAX_CARRIED_FRAGMENT = RAG_CHUNK_CHARACTER_LIMIT / 2;
const CONTINUES_SENTENCE = /[a-z,\-­]\s*$/;
const HYPHENATED_END = /\p{L}[-­]\s*$/u;

function bannerKey(line: string): string {
	return line.toLowerCase().replace(/\s+/g, ' ').replace(/\d+/g, '#').trim();
}

function findBanners(chunks: ExtractedChunk[], pages: Map<number, PageText>): Set<string> {
	const pageCounts = new Map<string, number>();
	for (const page of pages.values()) {
		const keys = new Set(edgeLines(chunks, page).map(bannerKey).filter(Boolean));
		for (const key of keys) pageCounts.set(key, (pageCounts.get(key) ?? 0) + 1);
	}

	const minPages = Math.max(MIN_BANNER_PAGES, Math.ceil(pages.size * BANNER_PAGE_RATIO));
	return new Set([...pageCounts].filter(([, count]) => count >= minPages).map(([key]) => key));
}

/**
 * Removes running headers and footers: lines at the top or bottom edge of a page that recur,
 * digits aside, on a large share of the document's pages.
 */
export function stripRepeatedPageLines(chunks: ExtractedChunk[]): ExtractedChunk[] {
	const pages = textChunksByPage(chunks);
	if (pages.size < MIN_PAGES_FOR_BANNERS) return chunks;

	const banners = findBanners(chunks, pages);
	if (banners.size === 0) return chunks;

	const cleaned = chunks.map((chunk) => ({ ...chunk }));
	for (const page of pages.values()) {
		cleaned[page.first].content = stripEdges(cleaned[page.first].content, banners, 'head');
		cleaned[page.last].content = stripEdges(cleaned[page.last].content, banners, 'tail');
	}
	return cleaned.filter((chunk) => chunk.chunkType !== 'TEXT' || chunk.content.trim());
}

/**
 * Moves the opening fragment of a page onto the previous page when a sentence runs across the
 * page break, so it lands in one chunk attributed to the page where it starts.
 */
export function joinCrossPageSentences(chunks: ExtractedChunk[]): ExtractedChunk[] {
	const pages = textChunksByPage(chunks);
	const joined = chunks.map((chunk) => ({ ...chunk }));
	const emptied = new Set<number>();

	for (const [pageIndex, page] of pages) {
		const next = pages.get(pageIndex + 1);
		if (!next) continue;

		const previous = joined[page.last];
		const following = joined[next.first];
		if (emptied.has(page.last)) continue;

		const previousText = previous.content.trimEnd();
		const followingText = normalizeWhitespace(following.content);
		if (!CONTINUES_SENTENCE.test(previousText)) continue;
		if (!/^\p{Ll}/u.test(followingText) && !HYPHENATED_END.test(previousText)) continue;

		const [fragment] = splitSentencesWithOffsets(followingText);
		if (!fragment || fragment.end - fragment.start > MAX_CARRIED_FRAGMENT) continue;

		previous.content = joinFragment(previousText, fragment.text);
		following.content = followingText.slice(fragment.end).trim();
		if (!following.content) emptied.add(next.first);
	}

	return joined.filter((_, index) => !emptied.has(index));
}
```

The rules in plain terms:

- **Banner:** a line is a banner when it is among the first or last two lines of a page and
  recurs, with digits ignored, on at least 3 pages and at least 30% of the document's pages. The
  check only runs on documents of 4 or more pages. Banners are deleted from page edges only.
- **Join:** when a page ends in a lowercase letter, comma or hyphen, and either the next page
  starts lowercase or the page ends with a hyphenated word, the next page's first sentence is
  moved back.
  The move is skipped when that sentence is over 600 characters. A page left empty is dropped.

### 1, 2 and 3: wiring in `src/lib/server/rag/chunk/text-extract.ts`

```diff
+import { joinCrossPageSentences, stripRepeatedPageLines } from './page-cleanup.ts';
 import { extractPageImages, type PageImage, type PdfDocument } from './page-images.ts';
@@
+function compactText(text: string): string {
+	return text.toLowerCase().replace(/\s+/g, '');
+}
+
+// pdf-parse finds tables in the same text layer getText reads, so a table whose every cell
+// already appears in the page text would only index that text a second time
+function isTableInPageText(table: TableArray, pageText: string): boolean {
+	const page = compactText(pageText);
+	const cells = table.flat().map(compactText).filter(Boolean);
+	return cells.length > 0 && cells.every((cell) => page.includes(cell));
+}
+
+function cleanUpPages(chunks: Chunk[], file: Source): Chunk[] {
+	const cleaned = stripRepeatedPageLines(chunks);
+	// Slides routinely end without punctuation, so a slide break is never a sentence break
+	return file.type === 'PPTX' ? cleaned : joinCrossPageSentences(cleaned);
+}
@@
 			for (const table of tablesByPage.get(pageNumber) ?? []) {
 				const content = tableToText(table);

-				if (content) {
+				if (content && !isTableInPageText(table, nativeText)) {
 					chunks.push({ chunkType: 'TABLE', source: file, pageIndex, content });
 				}
 			}
 		}

-		return { chunks, pageCount: textResult.total };
+		return { chunks: cleanUpPages(chunks, file), pageCount: textResult.total };
```

`extractText` also handles Word and PowerPoint files after they are converted to PDF, so these
cleanups reach those formats too. For PowerPoint, only the banner strip runs.

### 4: long OCR split in `src/lib/server/rag/chunk/chunker.ts`

```diff
 function chunkPage(page: ExtractedChunk): ParsedChunk[] {
 	const content = preparePageContent(page);
 	if (!content) return [];

+	if (page.chunkType === 'IMAGE' && content.length > RAG_CHUNK_CHARACTER_LIMIT) {
+		return chunkSentenceWindows(page, content);
+	}
+
 	if (page.chunkType === 'IMAGE') {
 		return [
```

On upstream `main`, each image's OCR text is one chunk regardless of length.

## Effect on the corpus

| Copy                                                 | TEXT  | IMAGE | Total |
| ---------------------------------------------------- | ----- | ----- | ----- |
| Upstream chunking (`runtime-pdf-eval-001`)           | 2,782 | 43    | 2,825 |
| With cleanups (`runtime-pdf-eval-001-step1-cleanup`) | 2,745 | 43    | 2,788 |

The 37 fewer TEXT chunks come from shorter pages, after banners are removed and fragments moved,
packing into fewer 1,200-character windows.

## Measured effect

These are two complete runs: the app default pipeline (union of semantic and BM25, reranked by
Ettin 32M), all 81 questions, page-level scoring at depth 10. Table recognition was off in both,
and only the chunker differs.

- Upstream chunking: `pdf-eval-default-p40-o20-all-20261116-180551-878808`
- With cleanups: `pdf-eval-default-p40-o20-all-20261116-161649-134275`

The table shows the change from upstream chunking to the cleanups. The p-values come from a
paired sign-flip test on per-query values.

| Method       | nDCG@10 (p)            | MRR@10 (p)                  | MAP@10          | Recall@10       |
| ------------ | ---------------------- | --------------------------- | --------------- | --------------- |
| BM25         | 0.7165 → 0.7108 (0.58) | 0.7077 → 0.6924             | 0.6521 → 0.6463 | 0.8529 → 0.8529 |
| **Semantic** | 0.6463 → 0.6686 (0.12) | 0.6130 → **0.6601 (0.027)** | 0.5772 → 0.6081 | 0.8056 → 0.7870 |
| Hybrid       | 0.7871 → 0.7902 (0.84) | 0.7810 → 0.7964 (0.42)      | 0.7274 → 0.7415 | 0.9053 → 0.8745 |

Semantic search put the first relevant page higher for 18 queries and lower for 9.

The next table splits the nDCG@10 change by question type.

| Method   | Fact (F, 42) | Topic, multi-page (T, 23) | Table (X, 16) |
| -------- | ------------ | ------------------------- | ------------- |
| BM25     | +0.001       | −0.026                    | +0.007        |
| Semantic | +0.020       | +0.035                    | +0.011        |
| Hybrid   | +0.015       | −0.022                    | +0.009        |

### All four configurations

Table recognition (Docling's Heron layout plus TableFormer) adds table chunks and leaves the page
text unchanged. Running it with and without the cleanups gives four corpus copies:

| Configuration                    | Semantic nDCG@10 | Semantic MRR@10 | Hybrid nDCG@10 | Hybrid MRR@10 | Hybrid Recall@10 |
| -------------------------------- | ---------------- | --------------- | -------------- | ------------- | ---------------- |
| Upstream chunking (baseline)     | 0.6463           | 0.6130          | 0.7871         | 0.7810        | 0.9053           |
| Cleanups                         | 0.6686           | 0.6601          | 0.7902         | 0.7964        | 0.8745           |
| Table recognition                | 0.6714           | 0.6397          | 0.7971         | 0.7971        | 0.9115           |
| **Cleanups + table recognition** | **0.6879**       | **0.6796**      | **0.8230**     | **0.8255**    | 0.9095           |

The next table gives each configuration's change against the baseline, with paired sign-flip
p-values.

| Configuration                | Semantic nDCG@10       | Semantic MRR@10        | Hybrid nDCG@10         | Hybrid MRR@10      |
| ---------------------------- | ---------------------- | ---------------------- | ---------------------- | ------------------ |
| Cleanups                     | +0.022 (p = 0.12)      | **+0.047 (p = 0.027)** | +0.003 (p = 0.84)      | +0.015 (p = 0.42)  |
| Table recognition            | +0.025 (p = 0.11)      | +0.027 (p = 0.16)      | +0.010 (p = 0.51)      | +0.016 (p = 0.43)  |
| Cleanups + table recognition | **+0.042 (p = 0.040)** | **+0.067 (p = 0.010)** | **+0.036 (p = 0.045)** | +0.045 (p = 0.051) |

- The cleanups and table recognition each improve semantic search by a similar amount, and the
  two gains roughly add up.
- Only the combination improves hybrid search: on its own, each one leaves hybrid within noise.
- The combination keeps hybrid recall (0.910 against 0.905), while the cleanups alone lose it.

Runs used: upstream `…-180551-878808`, cleanups `…-161649-134275`, table recognition
`…-docling-all-20261116-180013-146819`, both `…-docling-all-20261116-162235-429832` (all named
`pdf-eval-default-p40-o20-…`).

### What table recognition adds once the cleanups are in

This compares the two cleanup copies, with table recognition off and on. Only table recognition
differs. Table recognition doesn't change the page text, so every TEXT chunk is identical in
both copies.

- Cleanups: `pdf-eval-default-p40-o20-all-20261116-161649-134275`
- Cleanups + table recognition: `pdf-eval-default-p40-o20-docling-all-20261116-162235-429832`

| Method     | nDCG@10 (p)                 | MRR@10 (p)             | MAP@10 (p)              | Recall@10 (p)          |
| ---------- | --------------------------- | ---------------------- | ----------------------- | ---------------------- |
| BM25       | 0.7108 → 0.7112 (0.95)      | 0.6924 → 0.6929 (0.95) | 0.6463 → 0.6466         | 0.8529 → 0.8529        |
| Semantic   | 0.6686 → 0.6879 (0.14)      | 0.6601 → 0.6796 (0.22) | 0.6081 → 0.6260 (0.22)  | 0.7870 → 0.8086 (0.50) |
| **Hybrid** | 0.7902 → **0.8230 (0.025)** | 0.7964 → 0.8255 (0.11) | 0.7415 → 0.7753 (0.054) | 0.8745 → 0.9095 (0.12) |

The change in nDCG@10, split by question type:

| Method     | Table (X, 16)          | Fact (F, 42) | Topic (T, 23) |
| ---------- | ---------------------- | ------------ | ------------- |
| BM25       | +0.033 (p = 0.50)      | −0.012       | −0.001        |
| Semantic   | +0.106 (p = 0.13)      | 0.000        | −0.006        |
| **Hybrid** | **+0.155 (p = 0.017)** | 0.000        | +0.008        |

In the hybrid results, the first relevant page rose for 10 queries and fell for 5. The largest
moves are all table questions:

- X10 and X13 entered the top 10, at ranks 6 and 4.
- X16 went from rank 4 to rank 1.
- X1, X3 and X4 went from rank 2 to rank 1.

The losses are mostly one rank (F4, F28, T4, T8, X9), plus T18, which now sits at rank 10.

The same comparison without the cleanups, from `docs/DOCLING_TABLEFORMER_RETRIEVAL.md`:

| Table recognition's effect  | Without cleanups  | With cleanups          |
| --------------------------- | ----------------- | ---------------------- |
| Semantic nDCG@10            | +0.025 (p = 0.11) | +0.019 (p = 0.14)      |
| Semantic, table questions   | +0.135 (p = 0.09) | +0.106 (p = 0.13)      |
| Hybrid nDCG@10              | +0.010 (p = 0.51) | **+0.033 (p = 0.025)** |
| Hybrid, table questions     | +0.066 (p = 0.34) | **+0.155 (p = 0.017)** |
| Hybrid Recall@10            | 0.905 → 0.912     | 0.875 → 0.910          |
| Non-table questions, hybrid | −0.004            | +0.003                 |

- **Semantic search gains about the same either way.** The new table chunks are what the embedder
  matches, and the cleanups don't touch them.
- **Hybrid search gains three times as much with the cleanups.** Only with the cleanups is table
  recognition's hybrid gain significant on its own. Without them, table chunks competed with
  noisier page text for the reranker's slots, and hybrid stayed flat.
- **Table recognition wins back the recall the cleanups lost.** Hybrid Recall@10 returns from
  0.875 to 0.910, slightly above upstream's 0.905.

Why the cleanups help table chunks in hybrid search is a hypothesis, not tested. One explanation:
with banner lines gone, page chunks on unrelated pages score lower in the Ettin reranker, so
table chunks keep their top-10 slots.

### Reading the results

- **Semantic search benefits, and that is the cleanups' clearest effect.** The right page moves
  up the list (MRR +7.7%, p = 0.027). Banner lines such as `AFMAN11-2T-38V2 8 JUNE 2026 45` sit in
  nearly every page's first chunk, and they pull embeddings of unrelated pages toward each other.
  Removing them is the most plausible cause. This is a hypothesis; the banner strip and the join
  were not tested separately.
- **On their own, the cleanups don't carry over to hybrid search.** The Ettin reranker reorders
  the merged candidates, and BM25 already finds most of these pages. Hybrid nDCG@10 is flat and
  its Recall@10 drops by 3 points. BM25 also falls slightly on topic questions.
- **Part of the recall loss may come from how pages are scored, not from worse retrieval.** The
  join moves a sentence's tail onto the previous page, and `pdf-eval` credits a hit by its page
  number. A moved fragment can therefore be found on a page the question doesn't list. Multi-page
  topic questions, which have page ranges, are where BM25 and hybrid lost. This is unverified.
- **The cleanups help hybrid search only when table recognition is on.** There, they raise hybrid
  nDCG@10 by +0.026, against +0.003 without it. Most of that comes from the 16 table questions
  (X16, X7, X8 and X4 up; X13 down). The combined configuration's gain over the baseline is
  significant (p = 0.045), but that rests on a few large swings among 16 questions. A second
  corpus should confirm it before anyone relies on it.

## Risks of shipping them

- **Every PDF user sees the change.** The cleanups run on all PDFs, and on Word and PowerPoint
  files converted to PDF, with no setting to turn them off. Chunks, and so search results and
  citations, change for documents users have already seen in demos.
- **Existing libraries won't change until re-ingested.** Documents already in a user's database
  keep their old chunks until they are ingested again. Users would get mixed behavior.
- **Text can be lost.** A line counted as a banner is deleted. A real line that happens to recur
  at page edges on 30% of pages, such as a repeated table header or a form label, is removed
  from search.
- **Citations can shift.** A joined fragment is cited to the earlier page, which is correct for
  the sentence but differs from where the reader sees those words.
- **Tuning came from one small corpus.** The thresholds (2 edge lines, 30% of pages, 600-character
  fragments) were set while looking at these Air Force publications.

## Recommendation

The cleanups are promising **together with table recognition**. That combination is the only
configuration that beats upstream significantly, on both semantic and hybrid search, and it keeps
hybrid recall. On their own, the cleanups improve semantic ranking but not the default hybrid
search, and they cost recall. So don't turn them on by default by themselves. Next steps:

1. **Consider tying them to table recognition**, so they run only when that opt-in setting is on.
   That is the configuration that measured well, and default behavior stays as stakeholders know
   it.
2. **Measure each one alone.** Banner strip and sentence join each get a corpus copy, and the
   other two cleanups need a corpus where they actually fire: ruled tables for cleanup 3, long
   OCR for cleanup 4.
3. **Score topic questions with range-tolerant matching**, so moved fragments don't count as
   misses, before concluding the join hurts recall.
4. **Test on a second PDF corpus with more questions.** `pdf-eval` has 81 questions, and the
   significant results above rest partly on 16 table questions.
5. **Ship any cleanup behind a setting that is off by default**, as table recognition ships, or
   under that setting (step 1).

## Reproducing

The two corpus copies and both runs are kept locally:

```bash
ls .cache/beir/runtime-pdf-eval-001 .cache/beir/runtime-pdf-eval-001-step1-cleanup
npm run bench -- scores \
    pdf-eval-default-p40-o20-all-20261116-180551-878808 \
    pdf-eval-default-p40-o20-all-20261116-161649-134275 --markdown
```

To view the cleanup code: `git show 07891439:src/lib/server/rag/chunk/page-cleanup.ts`, and
`git diff d1f5ce11 HEAD -- src/lib/server/rag/chunk/text-extract.ts src/lib/server/rag/chunk/chunker.ts`.
That diff also shows the TableFormer and token-chunking changes, which are separate from these
cleanups.
