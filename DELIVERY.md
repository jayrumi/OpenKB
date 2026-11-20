# Delivery: OpenKB isolated PageIndex Flash adapter

**Follow-up:** the initial partial acceptance status below is superseded by `REPAIR_REPORT.md`: Codex SSE, required-page failure handling and loop-owned client cleanup are repaired, and real compilation → query/chat plus a browser Web UI check passed on an isolated copy. Production deployment is still not performed or authorized.

## Status

Implemented in `/home/jayrumi/openkb-development/flash-adapter`, branch `feature/pageindex-flash-codex`, based on upstream `ff54396e575ee6feb0113b631a34caa082b441cc`. **Not deployed.** Production source, Python environment, service, Web UI binding and network configuration were not modified. No existing book was reindexed.

**Acceptance is partial, not an end-to-end production success claim.** Actual Flash layout parsing and a Luna summary through existing subscription OAuth succeeded. Sol account/model access succeeded. The complete add/compile/query/Web UI orchestration is regression-tested with mocked model calls, not with a real full-book run. A separately approved, bounded staged end-to-end test is still required before production acceptance.

## 1. Changes and files

- `openkb/flash_adapter.py`: opt-in isolated process, exact SDK contract, PDF/config/SDK-keyed atomic locked cache, validated complete results, bounded wall timeout and process-group termination, safe stage/usage/retry logging.
- `openkb/flash_worker.py`: PageIndex 0.7.4 public local Flash submission/tree API in temporary isolated storage; worker-only version-pinned retry seam; conservative summary concurrency; no cloud fallback.
- `openkb/indexing_policy.py`: configurable timeout/retry/backoff; Retry-After-aware transient HTTP 429 handling; immediate hard-quota/auth abort; actual LiteLLM timeout handling.
- `openkb/indexer.py`: explicit backend switch, separate indexing model, legacy-compatible persistence of Flash trees/pages/PDFs, cleanup and existing-index reuse; legacy OAuth-aware client.
- `openkb/stage_config.py`, `config.py`: stage-specific model selection and layered configuration, preserving existing model fallback and literal identifiers.
- `agent/compiler.py`, `agent/query.py`, `agent/chat.py`, `cli.py`, `api.py`, `api_helpers.py`, `api_config.py`, `api_models.py`: compilation/query/chat/CLI/REST wiring and additive config schema.
- `openkb/provider_compat.py`: OAuth key-check compatibility and Responses transport/tool-choice repairs in application memory; installed dependency files remain untouched.
- `requirements-flash.txt`, `requirements-flash.lock`: separate exact-pinned worker environment; legacy `pyproject.toml` and `uv.lock` unchanged deliberately.
- New Flash, stage-model, Codex transport/legacy and rollback regression tests; additive existing API expectation update.
- `scripts/offline_test_guard.py`: rejects non-loopback network and unmocked inference during tests.
- `docs/pageindex-flash-codex.md`: configuration, authentication, security, deployment and rollback runbook.
- `docs/flash-evidence/`: captured baseline/final tests, dependency metadata, audit, structural/page retrieval evaluation and bounded inference results.

## 2. Actually installed and selected versions

Application development environment (same pinned SDK stack as production):
- PageIndex `0.3.0.dev3`
- LiteLLM `1.87.2`
- OpenAI Agents SDK `0.17.3`
- OpenAI Python SDK `2.44.0`

Isolated worker environment, installed and exercised on this Linux ARM Raspberry Pi:
- PageIndex `0.7.4` (latest released at investigation)
- LiteLLM `1.109.1` (deliberate vetted pin, not latest `1.110.1`)
- OpenAI Agents SDK `0.24.0`
- OpenAI Python SDK `3.51.0`

Python is 3.13. Both isolated environments pass `pip check`. Their pip is 26.2 after audit remediation; production tooling is untouched. The 79-entry worker lock records selected transitive versions; it is a version lock, not a full hash-enforced lock.

## 3. Example configuration

```yaml
# Preserve the existing value instead if desired: this is the fallback.
model: chatgpt/gpt-6.2-sol
indexing_model: chatgpt/gpt-6-luna
compilation_model: chatgpt/gpt-6.2-sol
query_model: chatgpt/gpt-6.2-sol
indexing_backend: flash
flash_python: /home/jayrumi/openkb-development/flash-venv/bin/python
indexing_concurrency: 1
indexing_timeout: 120
indexing_retries: 2
indexing_backoff: 2
flash_worker_timeout: 3600
# Optional Flash-only bounds, calibrate on a small representative document.
flash_max_calls: 30
flash_max_tokens: 50000
# Existing compilation parallelism knob, independently optional:
concurrency: 1
language: ru
```

Missing/null stage overrides inherit `model`. Existing resumed chat sessions retain their recorded model. The configuration example is not installed in any production KB. Flash call/token budgets do not cap subsequent knowledge compilation or chat.

## 4. Automated verification

- Clean original baseline: **1249 passed**, two existing unawaited-coroutine warnings.
- Final combined suite with actual isolated SDK test enabled: **1330 passed**, **0 skipped**, **3 warnings**, 31.48 seconds in the captured run.
- Warnings: two baseline coroutine-mock warnings and one upstream `fitz` deprecation from the actual SDK test.
- `ruff check .`: **All checks passed**.
- `mypy openkb`: **Success: no issues found in 61 source files** (existing notes about untyped function bodies remain).
- Independent review initially found two medium correctness defects (legacy OAuth client selection and the actual LiteLLM Timeout type). Both were fixed with regression tests. Focused independent re-review passed: 8 tests, no remaining security concerns or logic errors in that review scope.
- `git diff --check`: clean.
- Actual new SDK orchestration uses real PDF layout parsing/submission with networking forbidden and all model calls mocked.
- A three-page actual add-coordinator test forces compilation failure, checks rollback and survival of an existing document, then confirms successful-index cache reuse with only one mocked worker invocation across both attempts and the correct separate compilation model.
- Existing query/chat/streaming/API tests and additive stage routing/config tests pass.
- Development UI root plus its JS/CSS assets return HTTP 200 using FastAPI TestClient and the copied existing bundle. Production UI separately still returns HTTP 200; service PID remains 12586.

Not executed: real full-book import; real full add→compile→query tool conversation; live browser interaction with a deployed patched service; old-vs-new real-LLM indexing/answer-quality benchmark; production deployment/restart; full backup-restore rehearsal. These omissions are intentional and are not reported as passed.

## 5. Actual Codex Auth verification

No API key or paid substitute provider was used. Existing LiteLLM subscription OAuth was reused without manually copying or printing tokens.

Application smoke tests were explicitly bounded to **3 real inference requests**, **5551 provider-reported tokens** total (not the agent conversation/development token usage):
1. Exact `chatgpt/gpt-6-luna`: returned `OK`, 1639 tokens.
2. Exact `chatgpt/gpt-6.2-sol`: returned `OK`, 1639 tokens.
3. Actual `pageindex.flash.page_index_flash` with Luna on a small synthetic one-page technical PDF: one bounded summary call, 2273 tokens, 3.09 seconds, correct source-grounded summary and page range 1–1.

The first two tests capped output at 32 tokens, the Flash test at 96, with no provider retries and short request timeouts. No further real inference was performed by test code. The native Codex CLI nested-token file is not interchangeable with LiteLLM's existing flat OAuth store; direct native-file import is not implemented.

A separate zero-inference four-page technical fixture evaluation found the three expected section headings and page ranges, retrieved the expected facts with the actual OpenKB page-content tool and preserved page citation markers. It does not establish LLM answering quality or full-book speedup.

## 6. Risks and remaining limits

- Worker audit retains **one known vulnerable package**, SDK-required PyPDF2 3.0.1, CVE-2023-36464 / PYSEC-2026-1835. A hostile PDF can hang parsing; process timeout contains exposure but does not patch the parser. Use trusted PDFs. No blanket security-success claim is made.
- Legacy LiteLLM has public proxy-related advisories; this patch does not enable its proxy or prove OpenKB SDK-path exploitability. Broad application dependency upgrades are intentionally avoided.
- The retry policy plugs into private PageIndex helper seams inside the isolated, exact-version worker. SDK upgrades need contract/regression review, not a blind version bump.
- Complete successful-index checkpointing survives downstream compilation failure; mid-tree resume is not implemented. Layout-only documents without usable structure/OCR are not silently promoted to full success.
- Token limits are based on reported completed usage and can overshoot by the final response/in-flight calls. Subscription quotas are not equivalent to a simple token counter; compilation/chat can still exhaust them.
- Process-group cleanup is tested for errors/timeouts reaching the synchronous wrapper; async caller cancellation through `asyncio.to_thread` has not been proven to propagate to a worker.
- Concurrent OAuth refresh across two environments remains an operational risk; avoid parallel initial rollouts and preserve existing credential ownership.
- No measured old-vs-new speedup is claimed; the expensive old indexing path was not rerun against subscription quota.

## 7. Deployment and rollback

Detailed runbook: `docs/pageindex-flash-codex.md`.

**Before any production switch:** obtain explicit approval; make a fresh consistent full backup; retain the old source/environment and existing frontend bundle; install this patch side-by-side; run offline tests; use an unregistered copied KB for a newly bounded staged end-to-end verification. Only then switch the service executable during an approved maintenance window. Keep its host/port/Tailscale/SSH/OAuth configuration unchanged.

Development backup already created privately at:
`/home/jayrumi/openkb-development/backup-20261119T223442Z/`
It includes the KB/config archive, per-database SQLite backup-API snapshots with integrity checks and a SHA-256 manifest. Directory mode 700; archive mode 600. All 527 originally manifested production files remained byte-identical at verification; SQLite WAL/SHM sidecars can appear from normal live reads. This does not replace a fresh deployment-time snapshot.

Rollback first stops new workers/service and restores the previous executable and YAML settings. No existing-schema migration or automatic reindex occurred. If a data restore is necessary, restore wiki/raw/registry/metadata/blobs/databases as one coherent snapshot, saving post-snapshot additions first and obtaining permission before replacing data. Never delete/recreate OAuth credentials as a rollback shortcut.
