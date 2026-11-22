# Source-first compiler attribution gate

The compiler retrieves bounded, verbatim evidence from original
`wiki/sources/<document>.json` pages for summary, plan, concept and entity
generation. Common and per-target evidence windows are each bounded to 12,000
characters; a large book is not inserted wholesale into every model request.
Per-target retrieval uses the target slug/title; explicitly cited physical pages
are prioritized. Lexical ranking selects context, not semantic truth. Unrelated
zero-score windows do not fill remaining space when relevant matches exist.
Short text documents without page JSON use
the original text as one source unit (page 1); that is not a claim to recover PDF
pagination. Page JSON IDs are physical extraction page numbers, not printed
footer numbers. Invalid page JSON fails closed rather than substituting a summary.

## Selective cost policy

A conservative, configuration-independent routing heuristic looks for named
items in comma/semicolon or bullet enumerations in original source paragraphs.
Only candidate pages mentioning such names are routed. Descriptions as well as
bodies are included. The batch includes the actual summary to be published (or
already persisted PageIndex summary), not just an intermediate overview.
Having two entities is **not** a trigger by itself.

A routed document receives **one additional asynchronous compilation-model
request**, irrespective of candidate count. This uses the compilation credential
bundle and stage reasoning policy. There are no per-page audits, semantic
rewrites, or audit retry loops. The shared reasoning compatibility helper can
perform its narrowly authorized transport fallback; this is not a semantic retry.
Documents with no ambiguity signal preserve their existing request counts.

For a small source (at most 32 physical pages and 32,000 serialized source
characters), the audit receives **all original pages once**, shared across
candidates. `evidence_scope=complete_original_document_shared_once` declares
this mode; `candidate_pages` holds per-target page-key references into
`source_pages`, not duplicated text. This prevents broad summaries losing entity
definitions or access-control evidence merely because unrelated sections rank
higher lexically. It does not change the 12,000-character generation contexts.

Larger sources retain bounded per-target windows with explicit incomplete
coverage. Their `candidate_pages` values map physical page keys to exact selected
text; the shared page union is not extra evidence for other candidates. Checking
small-source eligibility does not first serialize an unbounded original book.

Evidence, all routed candidates and exact previous versions must fit a
100,000-character serialized batch bound, including audit instructions.
Candidate/baseline coverage is never silently truncated. An oversized complete
batch is refused, not silently downgraded to partial evidence. In selected-window
mode, insufficient support requires an unclear verdict and manual review.
Exceeding the bound requires
manual review and stops publication. This is a conservative application policy,
not a provider context-window guarantee; a smaller-context model can still reject
it, which also stops publication. A future configurable lower bound can be added
centrally; the gate must never mark uncovered targets as passed.

## Acceptance and publication

The audit is explicitly semantic: adjacent capabilities do not automatically
belong to a named product, while a genuine combined product can be supported.
General concept definitions need not all be facts attributed to this document.
The response must be strict JSON with exactly the requested target set, no extra
fields, duplicate targets, missing targets, or repaired/fenced JSON. Every target
must have a `supported` verdict and nonempty original-page quote evidence. Each
quote is whitespace-normalized and checked **on the exact physical page cited**.
Unknown pages, footer-number substitutions, unsupported/unclear verdicts,
malformed responses, truncation, and model errors all raise `GroundingError`
(a `RuntimeError`) before candidate pages, backlinks or success index entries
are written. Ambiguous candidates without original source pages also fail closed.
Evidence requires the current document slug, physical page and a verbatim quote.
Validation checks the exact original page and the evidence supplied for that
specific candidate: its allowed page references in complete mode, or its exact
windows in selected mode. Evidence from another candidate's different window
cannot be borrowed. Accepted citations retain `[source DOCUMENT p. N]`, so page 2
in two different documents is not ambiguous.

Updates include the exact previous page, including frontmatter and source
associations, in both generation and audit. The audit compares new/changed claims;
unchanged prior-source facts may remain without pretending they were revalidated
against the current PDF. Update verdicts must acknowledge that scope. A baseline
or old citation cannot support newly invented or strengthened claims.

All required page-generation tasks finish before the audit. Long-summary metadata
backfill is deferred until compilation succeeds. Existing add/recompile mutation
snapshots remain responsible for rollback of preexisting mutations performed by
the caller; raw documents and completed index caches are not reindexed by this
gate. A failed audit does not rewrite the candidate or retry with weaker standards.

## Limits (important)

This is risk reduction, **not deterministic semantic proof**. Enumeration/name
detection is lexical routing and can miss paraphrases, aliases, non-capitalized
names, non-Latin naming conventions, or claims outside enumerations, and can
route innocent prose. Quote validation establishes provenance, not entailment;
a model could still return a wrong supported verdict with a real but irrelevant
quote. The audit model is the semantic judge. Offline mock regressions establish
contract enforcement and transactional safety, not real-model attribution quality.
Real-model evaluation must be separately authorized; this implementation's tests
make no model, credential, or network calls.

The existing semantic linter was inspected: it is an interactive, post-publication
wiki audit with free-form output and up to 50 turns, not a source-page closed
batch validator. Reusing it would lose the prepublication, bounded-call and strict
coverage guarantees, so the focused gate lives in `openkb/grounding.py`.
