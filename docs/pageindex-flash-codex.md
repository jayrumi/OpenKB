# Isolated PageIndex Flash + Codex OAuth

## Deployment status and design

This is a development patch, not a deployed migration. Keep production running until a human explicitly approves deployment. The existing OpenKB Web UI/network bindings are unchanged.

Upstream OpenKB at `ff54396e575ee6feb0113b631a34caa082b441cc` pins PageIndex 0.3.0.dev3 and uses its collection/SQLite API. PageIndex 0.7.4 has incompatible constructors and filesystem storage, and requires newer LiteLLM/OpenAI/Agents SDK versions. Installing it over the legacy environment is not supported.

The opt-in Flash backend runs the new SDK in a separate Python process. Its validated tree is bridged into the old document format, preserving the existing wiki, source pages, images, registry, retrieval and mutation rollback path. No existing document is automatically reindexed. Successful indexing results are cached outside per-KB mutation snapshots so compilation failures can reuse them.

Flash's PDF layout parsing is local. Summaries and document description send text to the configured LLM provider. No PageIndex Cloud upload, API key or fallback paid provider is enabled by this adapter. Optimization/expansion is disabled by default to reduce inference.

## Environments and selected versions

- Legacy application: PageIndex **0.3.0.dev3**, LiteLLM **1.87.2**, OpenAI Agents **0.17.3**, OpenAI **2.44.0** (unchanged exact project pins).
- Isolated Flash worker: PageIndex **0.7.4**, LiteLLM **1.109.1**, OpenAI Agents **0.24.0**, OpenAI **3.51.0**.
- Worker direct requirements: `requirements-flash.txt`; complete selected transitive versions: `requirements-flash.lock`.
- Original `uv.lock` remains unchanged: it describes the legacy application, not the isolated worker. Never merge the two environments with `pip --no-deps` to bypass conflicts.

## Example per-KB configuration

Merge these keys into the destination's `.openkb/config.yaml` **only after deployment approval**. Do not replace the existing YAML wholesale.

```yaml
# Fallback for stages without a configured override; existing value is valid too.
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
# Optional hard per-indexing-run bounds. Start small on a sample, not a full book.
flash_max_calls: 30
flash_max_tokens: 50000
language: ru
```

Model identifiers are preserved. Stage overrides inherit the existing `model` when omitted/null. A successful short model probe proves account access, not full compilation/query correctness. Changing index settings affects new work/cache identity; it must not silently reindex existing registered documents.

## Authentication

Use the already configured LiteLLM ChatGPT OAuth store (`~/.config/litellm/chatgpt/auth.json`, or its existing `CHATGPT_TOKEN_DIR` / `CHATGPT_AUTH_FILE` override). No OpenAI API key is required. Do not print, copy into config, or check in credentials. A native Codex CLI auth file with nested `tokens` is not the same schema: pointing the LiteLLM token directory at it is not a supported import.

Real validation in this development task used the existing LiteLLM OAuth store and the exact IDs `chatgpt/gpt-6-luna` and `chatgpt/gpt-6.2-sol`; both returned `OK`. A separate single-call actual Flash summary also succeeded with Luna. Total real inference budget: **3 requests**, **5,551 provider-reported tokens**. No book-sized real indexing/compilation/query run was performed.

Two concurrent environments can race during OAuth token refresh. Avoid parallel indexing jobs during initial rollout and preserve the current credential owner/store. No new login or manual credential migration is part of this patch.

## Tests and reproducibility

Offline suite (mock provider calls; no subscription usage):

```sh
mkdir -p /home/jayrumi/openkb-development/test-home
HOME=/home/jayrumi/openkb-development/test-home \
LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONPATH=scripts \
/home/jayrumi/openkb-development/venv/bin/python -m pytest -p offline_test_guard -q
```

The guard rejects non-loopback network connections and unmocked LiteLLM completions. It is for automated regression tests, not real smoke tests. Do not rerun smoke tests as part of the default suite. Refer to the delivery report for final counts and any limitations.

Representative deterministic evaluation used a synthetic four-page technical PDF, actual Flash layout extraction, and actual OpenKB page-content retrieval. The three section headings, pages 2/3/4, source facts and page markers matched. This is not a measured LLM-answer-quality comparison or a demonstrated full-book speedup. Old-vs-new real indexing was not run because it would exceed the agreed bounded inference budget.

## Security and residual limits

- The known malicious LiteLLM releases 1.82.7/1.82.8 are not selected. Exact pins, PyPI release metadata/artifact digests and `pip-audit` were checked; hash matching is not proof of publisher trust.
- The worker's SDK still requires **PyPDF2 3.0.1**, with **CVE-2023-36464 / PYSEC-2026-1835** (crafted PDF can hang extraction). Its nominal fixed version refers to the successor `pypdf`; blindly pinning an unavailable PyPDF2 version is not a fix. Process-group termination and a wall-clock limit contain a hang, not the underlying vulnerability. Only ingest trusted files. A full sandbox/parser replacement remains future work.
- The isolated environments' pip was upgraded to 26.2 after initial audit findings. Production tooling was not upgraded.
- Existing legacy LiteLLM 1.87.2 has proxy-related public advisories; this patch does not deploy the LiteLLM Proxy or establish exploitability in OpenKB's SDK-only use. Broad provider upgrades are deliberately not bundled into this change.
- Flash local indexing requires a readable text layer; no OCR is introduced. Page-only/unreadable trees must not be silently reported as good hierarchical indexes.
- Complete-index checkpoint reuse is supported; arbitrary mid-request/mid-tree resume is not promised. Interrupted/failed attempts must not publish a partial successful document. Process-group cleanup is tested when a timeout/interrupt reaches the synchronous worker wrapper; cancellation propagation from an async caller waiting on `asyncio.to_thread` is not established by those tests.
- Conservative concurrency and retry limits reduce avoidable requests, but do not guarantee a subscription budget. A token cap checked after responses cannot undo the tokens consumed by the last response.

## Safe deployment procedure (NOT executed)

1. Review the branch diff, test report, security risks and model settings. Obtain explicit production deployment approval.
2. Take a fresh private backup of all registered KB directories, global/local config, registry, metadata, wiki, raw files and indexes. Use SQLite's backup API for consistent database snapshots. The development backup is not a substitute for a fresh deployment-time backup.
3. Build a **new application environment** from the legacy pins and this branch; build a separate worker environment from the exact worker lock. Retain the old environment and source untouched. Preserve the existing built `openkb/web` assets (or build the matching frontend); a Git-only clone omits those ignored assets.
4. Run the offline suite and compatibility tests in the new environment. Validate the staged service on a copied KB with loopback-only bindings; do not register synthetic KBs in production.
5. With explicit approval and a maintenance window, stop the application service, confirm no active mutations/jobs, take the final consistent backup, and switch only the executable/source environment. Do not change Tailscale, ports, SSH or OAuth stores.
6. Merge configuration keys into one approved KB. Start the service at its existing address. Verify existing document listing, original source hashes and Web UI before adding anything.
7. On a small approved PDF, verify indexing, summary/wiki compilation with the separate model, persisted page references, cited query answer and streaming Web UI. Set a new explicit inference budget for this test; this task's three-call budget is already used. Do not claim full end-to-end production acceptance until it passes.

## Rollback procedure

1. Stop the new service and its indexing workers; wait for mutation recovery rather than editing registry/SQLite files manually.
2. Restore the old executable/source environment and prior YAML settings (`indexing_backend` unset/legacy, prior `model`). Restart on the unchanged loopback binding.
3. The adapter preserves the old document schema and does not migrate existing indexes, so an application-only rollback should not require restoring the KB. Verify this on a copy first.
4. If data restoration is actually required, restore the **whole consistent snapshot** (wiki, raw files, registry, metadata, managed PDF blobs and SQLite snapshots together), not just `pageindex.db`. Save any post-backup additions separately first; a restore can otherwise erase them. Obtain explicit permission before replacing data.
5. Keep the external successful-index cache for diagnosis/reuse unless the user explicitly requests removal. Do not delete credentials or reauthorize as a rollback shortcut.
