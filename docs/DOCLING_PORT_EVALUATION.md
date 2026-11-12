# Docling port evaluation

Should Deployable Knowledge ship structure-aware PDF parsing, the partial TypeScript port of
Docling on branch `test/docling`? Measured 2026-11-12 and re-run from scratch the same day.

## Verdict

**Don't ship it. Keep the Step 1 cleanups to the legacy pipeline and shelve the layout pipeline
(Phase 1 and Phase 2).**

- **Search quality is a tie.** Over all 65 test questions, default (hybrid) search scores MRR
  0.857 with structured parsing and 0.841 with legacy. The 95% confidence interval on the
  difference runs from −0.047 to +0.082. No variant tried did better.
- **It is a trade, not an improvement.** Structured parsing is consistently worse at looking up a
  specific fact and better at finding the section that covers a topic. The two effects cancel out.
- **The cost is not a tie.** Ingest takes 1.6–1.8× as long. Users also need a 171 MB model
  download, and the app takes on a native canvas dependency (`@napi-rs/canvas`) that has never
  been through `electron:pack`. On top of that, about 1,650 lines of ported layout code would
  need maintaining.
- **Fixing tables would not change this.** Flattening tables to plain text fixed the worst
  example but left the overall fact-lookup gap where it was.
- **A retest against real Docling would not change the decision either.** See
  [What would change the decision](#what-would-change-the-decision).

## What was compared

| Run                        | What it does                                                                                                                                                                                                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Legacy (L)**             | pdf.js text, tesseract OCR for embedded images, and pdf-parse tables, cut into sentence windows of up to 1,200 characters. Includes the Step 1 cleanups: running headers and footers removed, duplicate tables dropped, sentences split by a page break joined.                                                           |
| **Structured P1**          | Every page is rendered and run through Docling's Heron layout model (ONNX). A port of Docling's layout post-processing and reading order follows. Chunks start at section headings; tables become their own chunks; page furniture and picture text are dropped.                                                          |
| **Structured P2**          | P1, with chunks also split at PDF bookmarks and a **section path** on each chunk (e.g. `Chapter 6 FORMATION > Section 6C—Tactical Formation > 6.28. Delayed 90-Degree Turns`). The path is prepended to the embedding input and the reranker passage, and indexed as a second BM25 column. This is what the branch ships. |
| **P2, content-only**       | P2 chunks, embedded without the path.                                                                                                                                                                                                                                                                                     |
| **P2, innermost section**  | P2 chunks, embedded with only the innermost section name.                                                                                                                                                                                                                                                                 |
| **P2, tables as text (T)** | P2, except Heron's table regions are emitted as plain text in reading order instead of being rebuilt row by row. A one-line change, used to test whether table reconstruction causes the losses.                                                                                                                          |

Not ported from Docling: TableFormer, docling-parse, the OCR/VLM stages, `HybridChunker` and
picture classification.

## Results

Two question sets, scored the same way. A result is a hit when it comes from the right document
and one of the right pages, so the score doesn't depend on how a pipeline cuts its chunks. The
tables give MRR@10: the average of 1 / rank of the first hit, counting 0 when there is no hit
in the top 10.

- **Fact lookups (42 questions):** each answer is one specific detail on one page, e.g. "How many
  days after the close-out date is an evaluation due to the MPF?".
- **Topic questions (23 questions):** each answer is a whole section, and any page of it counts,
  e.g. "How do I plan and build a low-level navigation route?".

### Hybrid search (the app's default: semantic + BM25, reranked)

| Run                   | Fact lookups (42) | Topic questions (23) | All 65 |
| --------------------- | ----------------- | -------------------- | ------ |
| Legacy                | **0.928**         | 0.682                | 0.841  |
| Structured P1         | 0.878             | 0.803                | 0.851  |
| Structured P2         | 0.877             | 0.820                | 0.857  |
| P2, content-only      | 0.841             | **0.855**            | 0.846  |
| P2, innermost section | 0.853             | 0.804                | 0.836  |
| P2, tables as text    | 0.865             | 0.820                | 0.849  |

### Semantic search only

| Run                   | Fact lookups (42) | Topic questions (23) | All 65 |
| --------------------- | ----------------- | -------------------- | ------ |
| Legacy                | **0.758**         | 0.664                | 0.725  |
| Structured P1         | 0.721             | 0.728                | 0.723  |
| Structured P2         | 0.645             | 0.841                | 0.714  |
| P2, content-only      | 0.714             | 0.754                | 0.728  |
| P2, innermost section | 0.704             | **0.862**            | 0.760  |
| P2, tables as text    | 0.662             | 0.841                | 0.725  |

BM25 alone scores within 0.02 of legacy on all 65 questions in every run.

### How sure we can be

The differences below are against legacy, with paired bootstrap 95% confidence intervals and
two-sided sign tests on rank-1 hits:

| Comparison           | Questions | Mode     | ΔMRR   | 95% CI           | Rank-1 gained / lost |
| -------------------- | --------- | -------- | ------ | ---------------- | -------------------- |
| P1 vs legacy         | Fact      | Hybrid   | −0.050 | −0.125 to +0.009 | 1 / 5                |
| P2 vs legacy         | Fact      | Hybrid   | −0.051 | −0.103 to −0.011 | 0 / 4                |
| P2 vs legacy         | Fact      | Semantic | −0.113 | −0.198 to −0.026 | 1 / 9 (p≈0.02)       |
| P1 vs legacy         | Topic     | Hybrid   | +0.121 | +0.035 to +0.225 | 2 / 0                |
| P2 vs legacy         | Topic     | Hybrid   | +0.139 | −0.026 to +0.292 | 4 / 2                |
| P2 vs legacy         | Topic     | Semantic | +0.176 | +0.058 to +0.304 | 6 / 0 (p≈0.03)       |
| P2 vs legacy         | All 65    | Hybrid   | +0.016 | −0.047 to +0.082 | —                    |
| Tables as text vs P2 | Fact      | Hybrid   | −0.012 | −0.052 to +0.032 | 1 / 2                |

**The runs themselves are not noisy.** Re-ingesting the corpus from scratch reproduced legacy and
P2 chunk for chunk and rank for rank. So all the uncertainty above comes from having only 65
questions, not from randomness in the pipeline.

## What the numbers mean

1. **Fact lookups favour legacy, in every structured variant.** In hybrid search, no structured
   run won more than one fact question that legacy lost, and each lost 4–7. That is too
   consistent to be luck.
2. **Topic questions favour structured.** Legacy misses three topic questions entirely in hybrid
   search (T1, T4, T18) that every structured run finds. Legacy's fixed-size sentence windows
   cut across section boundaries, so a heading and its body can land in different chunks.
   Structured chunks follow sections.
3. **Section paths move results in both directions.** Prepending the path to the embedding
   costs semantic search 8 rank-1 hits on fact lookups and gains it 6 on topic questions. The
   path makes every chunk in a section embed alike. That helps when the question is about the
   section and hurts when the question is about one paragraph in it. Neither embedding choice
   wins on both sets, which is why option (a) in the old handoff (embed content only) is not a
   fix. It just moves the loss from one question type to the other.
4. **Tables are a real defect but not the main cause.** Without TableFormer, a table cell that
   wraps across lines is split across rows. On the T-38 gradesheets (afman11-2t-38v2, where
   Heron labels 60 of 66 pages as tables), that shreds sentences, so "Slow flight: Airspeed −3
   to +5 knots" becomes fragments in different rows. Emitting tables as plain text fixed that
   question (semantic rank 8 → 3) but moved fact-lookup MRR by only −0.012. Most of the
   fact-lookup gap is on ordinary text pages. It comes from the chunking itself: structured
   chunks are smaller (729 characters on average against 918) and bounded by sections. That
   changes which chunk the reranker puts first.
5. **Which pipeline is better depends on what users ask.** With hybrid search, structured
   overtakes legacy once roughly 27–29% of questions are topic-style rather than fact lookups.
   The topic-side uncertainty is wide, so the real crossover could be well above or below
   that. Nobody has measured what this app's users actually ask.

## Ingest cost

| Measurement                                          | Legacy  | Structured      |
| ---------------------------------------------------- | ------- | --------------- |
| Full corpus, 1,071 pages, idle machine (re-run)      | 570 s   | 891 s (1.56×)   |
| Benchmark, 6 PDFs, 889 pages, extraction + embedding | 400.8 s | 711.1 s (1.77×) |
| Chunks produced (full corpus)                        | 2,788   | 3,386           |

- **Layout analysis costs a steady 0.36–0.44 s per page** on 4 inference threads. That is
  roughly as long as embedding the page takes.
- **Short documents suffer most:** afman11-2t-38v2 takes 16 s legacy and 43 s structured.
- **The app stays responsive.** The worst event-loop stall was 207 ms structured and 496 ms
  legacy.
- **Phase 2 adds no measurable time over Phase 1.**

## What would change the decision

- **Evidence that users mostly ask topic-style questions.** If well over a third of real queries
  are "what are the rules for X" rather than "what is the value of Y", structured parsing would
  probably win. The way to find out is a set of questions written by someone who uses these
  documents, not generated from the pages.
- **Bookmark-aware legacy chunking (untested, cheap).** The topic gains come from chunks that
  follow sections. These AF publications have usable PDF bookmarks, and the branch can already
  read them (`layout/outline.ts`) without the layout model. Starting a new legacy chunk at each
  bookmark could capture most of the topic gain with no model download and no extra ingest
  time. It is the only follow-up worth trying, and it only matters if topic questions matter.
- **Real Docling (docling-serve) is not worth retesting for this decision.** It is Python, so the
  app can't ship it. Its main advantage over this port is TableFormer, and the tables-as-text
  run shows table quality is not what limits results. Its `HybridChunker` also prepends headings
  to chunks, so it would face the same trade-off as point 3.

## If the branch is shelved

- **Keep:** `page-cleanup.ts` (running headers and footers, cross-page sentence joins, stranded
  headings), the duplicate-table drop in `text-extract.ts`, and the long-OCR split in
  `chunker.ts`. They run on the legacy path and are in the legacy numbers above.
- **Remove:**
  - `src/lib/server/rag/chunk/layout/`
  - the `structured_pdf_parsing` setting, its endpoints, service, store and Settings section
  - the `section_path` column and FTS column
  - `chunk-passage.ts`
  - the `.cache/layout` registration
  - the Docling and Heron entries in `THIRD-PARTY-NOTICES.md`
  - the User Guide and `DOCUMENTS_AND_SEARCH.md` text describing them
- **Migrations 0006 and 0007 are uncommitted.** Delete them along with the schema changes rather
  than shipping and reverting them.

## Setup

- **Machine:** 16 threads, Linux (WSL2). The layout worker uses 4 inference threads.
- **Models:** embedding `nomic-embed-text-v1.5.Q8_0.gguf` (llama.cpp), reranker `ettin-32m`.
- **Corpus:** 8 PDFs, 1,071 pages: afi36-2406 (302), aetcman11-251 (203), dafman36_2806 (231),
  dafi36-2903 (165), afman11-2t-38v3 (59), afman11-2t-38v2 (66), dafi90-160 (17),
  BrownBook (28).
- **Harness:** each run ingests the corpus with the app's own `ingestDocument` into a fresh
  database. It then queries with `searchSemantic`, `searchBm25` and `searchHybrid` (top 10).
  Legacy and P1 used a source snapshot taken before Phase 2.
- **Hybrid search** takes the top 20 semantic and top 20 BM25 results and lets the cross-encoder
  reranker order them. So its rank 1 is the reranker's choice.

## Caveats

- **Claude wrote both question sets.** The fact questions came from reading single pages, which
  favours legacy. The topic questions were written afterwards, to test whether structure helps.
  Their page ranges come from the PDF bookmarks, and they favour structured. Neither set comes
  from real usage. The 42 : 23 mix is arbitrary, and the "All 65" column depends on it. Read
  the two sets separately.
- **Single questions are fragile.** On T1, legacy and P1 produce identical chunks for the
  target page, yet hybrid ranks it "not found" for legacy and 10th for P1. Which chunks rank
  above it depends on the rest of the corpus. Only the patterns across many questions carry
  weight.
- **This port is not Docling.** It follows Docling's source but was never compared box for box
  with Docling's output.

## Appendix: questions and ranks

Rank of the first hit; — means not in the top 10. Columns: L = legacy, P1, P2, T = P2 with
tables as text.

### Fact lookups

| #   | Question                                                                                               | Document        | Page(s) | Semantic L / P1 / P2 / T | Hybrid L / P1 / P2 / T |
| --- | ------------------------------------------------------------------------------------------------------ | --------------- | ------- | ------------------------ | ---------------------- |
| F1  | How many days after the close-out date is an evaluation due to the MPF?                                | afi36-2406      | 12      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F2  | Who has to make sure nobody in a rating chain is related to the member being rated?                    | afi36-2406      | 25      | — / — / — / —            | 1 / 1 / 2 / 2          |
| F3  | Can a performance feedback session be done over the phone?                                             | afi36-2406      | 60      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F4  | Is an annual evaluation required for someone retiring before the next static close-out date?           | afi36-2406      | 110     | 1 / 1 / 2 / 2            | 1 / 3 / 2 / 3          |
| F5  | When does the first annual evaluation close out for an Air Force Band member promoted to TSgt?         | afi36-2406      | 160     | 1 / 1 / 4 / 4            | 2 / 2 / 3 / 1          |
| F6  | Is there a minimum number of days of supervision for a higher level reviewer?                          | afi36-2406      | 24      | — / — / — / —            | 1 / 1 / 1 / 1          |
| F7  | How long does a ratee have to submit a rebuttal on a referral evaluation?                              | afi36-2406      | 41      | 2 / 2 / 2 / 2            | 1 / 1 / 1 / 1          |
| F8  | Can an officer evaluation recommend the ratee for promotion to the next grade?                         | afi36-2406      | 85      | 4 / 3 / 3 / 3            | 1 / 1 / 1 / 2          |
| F9  | How late can a reserve officer's PRF be changed before the central selection board meets?              | afi36-2406      | 230     | 1 / 1 / 2 / 1            | 1 / 2 / 2 / 2          |
| F10 | What does the military personnel flight do for the performance feedback program?                       | afi36-2406      | 60      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F11 | Who is the aircrew's safety observer on the ground before flight?                                      | aetcman11-251   | 15      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F12 | At what angle of attack does light buffet occur?                                                       | aetcman11-251   | 40      | 2 / 2 / 2 / 2            | 2 / 2 / 2 / 2          |
| F13 | What does the wingman do when lead starts a delayed 90 degree turn into the wingman?                   | aetcman11-251   | 90      | 4 / 1 / 1 / 1            | 3 / 1 / 2 / 2          |
| F14 | Which map should I use for a VFR low-level route below 6,000 feet AGL?                                 | aetcman11-251   | 150     | 4 / 4 / 4 / 4            | 1 / 1 / 1 / 1          |
| F15 | Who is ultimately responsible for the safe conduct of a formation flight?                              | aetcman11-251   | 62      | 1 / 5 / 7 / 7            | 1 / 1 / 1 / 1          |
| F16 | What should a wingman do if they get spatially disoriented in formation?                               | aetcman11-251   | 125     | 1 / 2 / 2 / 2            | 1 / 1 / 1 / 1          |
| F17 | How far away should the chase aircraft stay during a controlled ejection?                              | aetcman11-251   | 125     | 1 / 1 / 2 / 2            | 1 / 1 / 1 / 1          |
| F18 | What is heavy buffet?                                                                                  | aetcman11-251   | 40      | 3 / 3 / 2 / 2            | 2 / 2 / 2 / 2          |
| F19 | Can someone get two personal decorations for the same achievement?                                     | dafman36_2806   | 20      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F20 | Should an Airman in a joint billet get a DAF decoration or a joint decoration?                         | dafman36_2806   | 20      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F21 | How is wounded in action defined for awards?                                                           | dafman36_2806   | 60      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F22 | Is a long period of combat duty enough for a Presidential Unit Citation?                               | dafman36_2806   | 120     | 4 / 2 / 2 / 2            | 1 / 1 / 1 / 1          |
| F23 | Can retired or separated veterans still ask to be considered for a decoration?                         | dafman36_2806   | 32      | 1 / 1 / 2 / 2            | 1 / 1 / 1 / 1          |
| F24 | How soon after approval must decoration certificates and citations be distributed?                     | dafman36_2806   | 95      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F25 | Are gold caps on teeth allowed just for looks?                                                         | dafi36-2903     | 45      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F26 | What can pregnant Airmen wear during unit PT?                                                          | dafi36-2903     | 80      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F27 | Does an Airman wearing tribal regalia at an outdoor ceremony also have to wear military headgear?      | dafi36-2903     | 130     | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F28 | Which badges are mandatory to wear on the uniform?                                                     | dafi36-2903     | 62      | — / — / — / —            | 7 / 3 / — / —          |
| F29 | What sock colors are allowed with the PT uniform?                                                      | dafi36-2903     | 100     | 1 / 2 / 1 / 1            | 1 / 5 / 1 / 1          |
| F30 | What is the minimum altitude for a lead change over water in a T-38 formation?                         | afman11-2t-38v3 | 15      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F31 | Which form is used to log unplanned supersonic flights?                                                | afman11-2t-38v3 | 35      | 1 / 2 / 1 / 1            | 1 / 1 / 1 / 1          |
| F32 | What slow flight airspeed tolerance earns a qualified grade for aircraft handling on a T-38 checkride? | afman11-2t-38v2 | 30      | 2 / 6 / 8 / 3            | 1 / 2 / 2 / 2          |
| F33 | Should the instrument checkride approach be flown somewhere other than the pilot's home field?         | afman11-2t-38v2 | 7       | 1 / 1 / — / —            | 1 / 1 / 1 / 1          |
| F34 | Do pilots with multiple qualifications need a separate T-38 instrument evaluation?                     | afman11-2t-38v2 | 7       | 2 / 2 / 2 / 2            | 1 / 1 / 1 / 1          |
| F35 | How many days does a commander have to send an approved waiver to the higher headquarters OPR?         | dafi90-160      | 6       | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F36 | What should a unit do if it gets an unauthorized change to a publication?                              | dafi90-160      | 6       | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F37 | What statement has to appear in the header of every directive publication?                             | dafi90-160      | 11      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F38 | What should an Airman First Class focus on?                                                            | BrownBook       | 14      | 1 / 2 / 2 / 2            | 1 / 1 / 1 / 1          |
| F39 | Who is the commander's key enlisted advocate at wing level and above?                                  | BrownBook       | 19      | — / — / — / —            | 1 / 2 / 1 / 2          |
| F40 | What do enlisted career field managers do?                                                             | BrownBook       | 19      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| F41 | What is the focus of the enlisted evaluation system?                                                   | BrownBook       | 24      | 2 / 1 / 2 / 2            | 1 / 1 / 1 / 1          |
| F42 | What are the three components of the Total Force?                                                      | BrownBook       | 9       | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |

### Topic questions

| #   | Question                                                                                          | Document        | Page(s) | Semantic L / P1 / P2 / T | Hybrid L / P1 / P2 / T |
| --- | ------------------------------------------------------------------------------------------------- | --------------- | ------- | ------------------------ | ---------------------- |
| T1  | What guidance is there for handling emergencies and abnormal situations during takeoff and climb? | aetcman11-251   | 23–24   | 1 / 2 / 1 / 1            | — / 10 / 3 / 3         |
| T2  | How do you fly the traffic pattern and land with one engine shut down?                            | aetcman11-251   | 38–39   | 1 / 1 / 1 / 1            | 2 / 1 / 1 / 1          |
| T3  | How should a formation handle engine start, check-in and taxi before takeoff?                     | aetcman11-251   | 67–68   | 5 / 5 / 4 / 4            | 1 / 1 / 1 / 1          |
| T4  | How does a wingman recover after closing too fast and passing the lead aircraft?                  | aetcman11-251   | 83–84   | 4 / 8 / — / —            | — / 1 / 2 / 2          |
| T5  | How do I plan and build a low-level navigation route?                                             | aetcman11-251   | 159–161 | 2 / 1 / 1 / 1            | 4 / 5 / 2 / 2          |
| T6  | How do I manage and correct my timing to hit the time on target on a low-level?                   | aetcman11-251   | 166–168 | 1 / 1 / 1 / 1            | 1 / 1 / 2 / 2          |
| T7  | What should a low-level formation do when a wingman loses sight of the other aircraft?            | aetcman11-251   | 175–176 | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T8  | How many evaluators does each type of evaluation form need, and who can serve as one?             | afi36-2406      | 20–21   | 2 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T9  | What happens to a feedback form after the session, and who is allowed to see it?                  | afi36-2406      | 61–62   | 2 / 1 / 1 / 1            | 4 / 6 / 5 / 5          |
| T10 | In what situations is a training report written instead of a normal evaluation?                   | afi36-2406      | 175–176 | — / — / 4 / 4            | 5 / 2 / 2 / 3          |
| T11 | How is the evaluation correction board set up and how does it process requests?                   | afi36-2406      | 254–257 | 1 / 1 / 1 / 1            | 1 / 1 / 3 / 2          |
| T12 | What are typical reasons for appealing an evaluation and what evidence is needed for each?        | afi36-2406      | 295–298 | 4 / 3 / 1 / 1            | 1 / 1 / 1 / 1          |
| T13 | What maps and charts do T-38 pilots need to prepare for a flight?                                 | afman11-2t-38v3 | 6–7     | — / — / 1 / 1            | 1 / 1 / 1 / 1          |
| T14 | What should a T-38 pilot do after losing all radio communication?                                 | afman11-2t-38v3 | 27–28   | 3 / 3 / 2 / 2            | 7 / 2 / 1 / 1          |
| T15 | What shoes and socks may be worn with service and dress uniforms?                                 | dafi36-2903     | 94–95   | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T16 | What special uniform items are authorized for Security Forces defenders?                          | dafi36-2903     | 118–120 | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T17 | What is the process for creating or changing an occupational badge or patch?                      | dafi36-2903     | 143–144 | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T18 | What achievements qualify someone for the Air Medal?                                              | dafman36_2806   | 77–78   | 2 / 4 / 1 / 1            | — / 2 / 1 / 1          |
| T19 | Who is eligible for the National Defense Service Medal?                                           | dafman36_2806   | 161–162 | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T20 | What service qualifies for the Vietnam Service Medal?                                             | dafman36_2806   | 223–225 | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T21 | How should the citation for a decoration be prepared?                                             | dafman36_2806   | 88–89   | 4 / 1 / 3 / 3            | 3 / 2 / 1 / 1          |
| T22 | What does teamwork and partnership with other organizations look like for enlisted Airmen?        | BrownBook       | 12      | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
| T23 | What is the difference between mentoring and coaching for Airmen?                                 | BrownBook       | 23–24   | 1 / 1 / 1 / 1            | 1 / 1 / 1 / 1          |
