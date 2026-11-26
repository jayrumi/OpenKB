# Physical PDF source links

The UI recognizes `[source <doc_name> p. <physical-page>]` citations in wiki and
chat Markdown, and explicit `pages` on `get_page_content` / `read_document`
trace entries (including restored sessions). Pages are **1-based physical PDF
pages**, not printed page labels. Missing/ambiguous originals, unknown pages,
and malformed page specifications remain non-clickable text. No page 1 fallback
is invented. Page specifications expand to at most 32 individual links.

## Read-only API

- `POST /api/v1/document/pdf/resolve` with
  `{"kb":"<name>","doc_name":"<registered name>","page":11}` returns
  `{hash, doc_name, page, pages}` only for a unique registered, readable PDF and
  an in-range physical page.
- `GET /api/v1/document/pdf?kb=<name>&hash=<sha256>&page=11` returns the original
  bytes with `application/pdf`, inline disposition, `nosniff`, `no-store`, and
  HTTP byte-range support. The page parameter is validated; the UI appends
  `#page=11` to the viewer URL to select the physical page.

Both endpoints use the existing opt-in bearer authorization and KB-name resolver.
No filesystem path is accepted from the request. A hash must belong to the selected
KB registry. Only its `raw_path` (or legacy `path`) resolving to an existing PDF
**inside that KB's raw directory** is served. Symlinks/traversal outside that
boundary, wiki/config files, cloud-only sources, and external originals are not
served. This does not introduce per-KB user ACLs beyond OpenKB's existing auth model.

The browser fetches bytes with an Authorization header and gives the native PDF
viewer a local blob URL plus `#page=N`; no credential is placed in a URL. The modal
also offers a direct original-PDF page link for opening in another tab. Blob URLs
are revoked on source unmount / KB change. Each chat turn keeps its original KB,
even when the current KB dropdown changes.

## Offline verification

```sh
LITELLM_LOCAL_MODEL_COST_MAP=true /home/jayrumi/openkb-development/venv/bin/python -m pytest \
  tests/test_pdf_sources.py tests/test_api.py tests/test_api_documents.py \
  tests/test_markdown_renderer.py tests/test_file_size.py -q
cd frontend
npm run test:pdf   # requires local Chromium; override path with CHROMIUM
npm run build
```

The browser regression uses explicit synthetic responses to test React behavior,
missing sources, bearer headers, and physical-page blob URLs without inference.
The separate acceptance evidence under `v6-hardening/pdf-source-links/` verifies
the real isolated 13-page original PDF, wiki citation, and restored PageIndex chip
in Chromium. Safari/mobile native-PDF handling remains platform-specific; the
external-tab link is provided as a fallback. No reindex, model call, deployment,
or production KB mutation is needed.
