# Oversized PageIndex summary attribution audit

This extends only the attribution gate. It does not regenerate an index, change
models, relax semantic validation, or publish a partial result.

## Admission and coverage

`compile_long_doc` probes the existing singleton request builder locally. Only a
typed `AuditSizeError` admits a stored summary artifact. Ordinary whole-target
requests, exact previous-version comparison, multisource provenance, prompt and
response schema, source selection, and the 100,000-character request bound are
unchanged. Non-size probe refusals are deferred to the existing gate, not split.

The artifact loads the existing default-collection tree by exact document ID using
SQLite `mode=ro`, without constructing an index client. It checks the database
filename's stem against the separate wiki slug. The candidate must be exactly
`description + newline + frontmatter + existing recursive renderer output`.
The frontmatter must identify the matching PageIndex source. Existing annotations,
ambiguous headings, duplicate node IDs, malformed ranges, non-nested ancestor
ranges, stale trees, and unrendered text fail closed. This is not a general Markdown
chunker.

One request owns each rendered section's **own** span, excluding descendants.
The first span also owns the exact description/frontmatter preamble. Every later
request repeats that preamble and each ancestor's exact own section as candidate
context, never as source evidence. All owned spans must cover the full parent
contiguously, exactly once; concatenating their UTF-8 bytes reproduces the parent.
A fragment key binds the parent, exact character offsets, and owned UTF-8 SHA-256.
Frozen target/source/tree digests, exact snapshots, and a context/span manifest
bind the plan. On-disk source/summary byte hashes and the exact stored tree JSON
are read back before acceptance. Missing or changed inputs refuse publication.

The verified oversized parent is required once; fragments never pass through
lexical risk exemptions. Every fragment is planned before audit inference. There
is no subtree grouping, text truncation, node omission, or semantic retry. A
nonempty prior-version baseline cannot be projected onto fragments; an oversized
singleton with such a baseline refuses locally. An individual oversized section
also refuses locally.

Evidence remains the existing original physical pages / selected windows (12,000
characters for window selection), never generated ancestors. Supported results
still require quotes contained in both the fragment's own allowed windows and the
original page. Unsupported, unclear, missing, duplicate, wrong-key, malformed, or
out-of-window results are fatal. Retrieval and structural coverage are **not**
semantic proof; insufficient evidence must still produce refusal.

All fragments and other required candidate audits must pass before any staged
candidate gets an annotation. Final accepted fragment IDs must match the exact
expected set, not overwrite a parent key. Their citations are deduplicated into
**one** standard parent-level `Source attribution evidence:` suffix. The original
body is unchanged; existing compiler backlinks and deferred metadata handling
remain allowed. This does not promise a byte-identical published file after those
annotations. Existing mutation coordination protects wiki/index writes.

## Offline verification

Run from the development checkout with the application venv, isolated HOME and
its existing offline inference/network guard:

```sh
mkdir -p /home/jayrumi/.hermes/cache/scratch/section-audit-test-home
HOME=/home/jayrumi/.hermes/cache/scratch/section-audit-test-home \
PYTHONPATH=scripts \
/home/jayrumi/openkb-development/venv/bin/pytest -p offline_test_guard -q
/home/jayrumi/openkb-development/venv/bin/ruff check .
/home/jayrumi/openkb-development/venv/bin/mypy openkb
```

The opt-in actual-book test only reads the retained KB. Its optional report is a
new output, not an edit to retained captures or feasibility probes:

```sh
HOME=/home/jayrumi/.hermes/cache/scratch/section-audit-test-home \
PYTHONPATH=scripts \
OPENKB_SECTION_AUDIT_BOOK=/home/jayrumi/openkb-development/book-unlimited-20261124-r2/offline-diagnosis/recovered-kb \
OPENKB_SECTION_AUDIT_PROOF=/home/jayrumi/openkb-development/summary-section-audit-proof.json \
/home/jayrumi/openkb-development/venv/bin/pytest \
  -p offline_test_guard tests/test_summary_audit_book.py -q -s
```

It independently plans all 949 requests using the existing source selector, checks
exact source-window containment, request/evidence bounds, and UTF-8 reassembly.
It then reuses that exact plan for two explicitly synthetic verdict runs: final
fragment refusal (zero mutations) and all-pass (one suffix, original body intact).
No semantic support is claimed. The test blocks socket connections; the guard
also prohibits real LiteLLM calls. The report includes every fragment's span,
owned/context digests and selected physical page IDs, plus input hashes.

## Compilation-only replay after independent acceptance

Do **not** execute a paid continuation merely because the offline tests pass.
Do not run the historical `compile_diagnostic.py` unchanged: its stage name is
already occupied and its runtime/accounting and captured inputs are protected.

The parent should prepare a **new private stage/copy** of the recovered KB and
retain the existing diagnostic's `kb_ingest_lock` → `recompile_mutation` → private
capture → owned-LLM-loop boundary. Replay the retained actual overview and plan
responses only after asserting full application-message equality with their saved
requests (including cached text blocks); never substitute synthetic plans or a
compact overview for the original full-index generation context. Preserve existing
941 indexing calls and all historical accounting; use the already-approved meter
without editing the runtime, ledgers, caps or latches in this implementation.

Inside that protected boundary, the existing compilation coroutine is:

```python
run_owned_llm_loop(compile_long_doc(
    "Instrumentariy_hraneniya_i_analiza_dannyh",
    new_stage_kb / "wiki/summaries/Instrumentariy_hraneniya_i_analiza_dannyh.md",
    "49d41265-4b10-449c-87bd-54ef0ada15c4",
    new_stage_kb,
    "openai/responses/gpt-6-sol",  # existing diagnostic model; do not change
    doc_description=preserved_tree["doc_description"],
    max_concurrency=1,
))
```

Before admitting paid calls, verify original summary SHA-256
`cf8a0c4a8948c65fb93fdf34de36fb3ac937fdd80b2b697d2aae5f1d273ba5b3`,
949 stored/rendered sections and 648 nonblank physical source pages. Keep indexing
entry points forbidden. The DB document filename retains `.pdf`; the coroutine
uses the extensionless slug above. Do not invoke add, reindex, all-doc recompilation,
schema refresh, deployment, or a success-registry shortcut. Any audit refusal is
terminal: retain capture and rollback, with no model-retry/regeneration loop.
Accept published results separately only after full audits succeed and original
body, one evidence suffix, permitted backlinks, and protected input/accounting
invariants are checked.
