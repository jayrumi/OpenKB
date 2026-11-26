"""Read-only, registry-authorized physical PDF sources (no arbitrary paths)."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO

import anyio
import pymupdf
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from openkb.api_helpers import _resolve_kb, require_bearer_token
from openkb.state import HashRegistry

pdf_router = APIRouter()


class PdfResolveRequest(BaseModel):
    kb: str = Field(min_length=1)
    doc_name: str = Field(min_length=1)
    page: int = Field(ge=1, strict=True)


class _SnapshotResponse(FileResponse):
    """Reuse Starlette ranges, but never reopen the original or use pathsend."""

    def __init__(self, snapshot: BinaryIO, filename: str, digest: str):
        self.snapshot = snapshot
        super().__init__(
            "",  # No delivery pathname: all reads go through _open_file.
            stat_result=os.fstat(snapshot.fileno()),
            media_type="application/pdf",
            filename=filename,
            content_disposition_type="inline",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
                "ETag": f'"{digest}"',
            },
        )

    @asynccontextmanager
    async def _open_file(self) -> AsyncIterator[anyio.AsyncFile[bytes]]:
        yield anyio.wrap_file(self.snapshot)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope = dict(scope)
        scope["extensions"] = {
            key: value
            for key, value in scope.get("extensions", {}).items()
            if key != "http.response.pathsend"
        }
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Also runs on malformed/unsatisfiable ranges, disconnect and send errors.
            self.snapshot.close()


def _pdf(kb_dir: Path, meta: dict, page: int, digest: str) -> tuple[BinaryIO, int, str]:
    """Hash, parse and serve one private disk snapshot, not a mutable original.

    PyMuPDF's filename API avoids loading the whole PDF into Python memory.
    On this Linux server /proc/self/fd pins the anonymous temporary file's inode;
    no named temporary file or original pathname is reopened for validation.
    Ownership of the returned handle transfers to the caller.
    """
    raw = kb_dir.resolve() / "raw"
    stored = meta.get("raw_path") or meta.get("path")
    if not isinstance(stored, str) or not stored:
        raise HTTPException(404, "Original PDF unavailable.")
    snapshot = None
    try:
        target = (kb_dir / stored).resolve()
        if (
            not target.is_relative_to(raw)
            or target.suffix.lower() != ".pdf"
            or not target.is_file()
        ):
            raise ValueError("Original PDF unavailable")
        snapshot = tempfile.TemporaryFile(mode="w+b")
        with target.open("rb") as source:
            while chunk := source.read(64 * 1024):
                snapshot.write(chunk)
        snapshot.flush()
        snapshot.seek(0)
        # Hash the snapshot itself, not a second view of the original.
        actual = hashlib.sha256()
        while chunk := snapshot.read(64 * 1024):
            actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError("Original PDF changed")
        snapshot.seek(0)
        if snapshot.read(5) != b"%PDF-":
            raise ValueError("Not a PDF")
        with pymupdf.open(f"/proc/self/fd/{snapshot.fileno()}") as doc:
            if not doc.is_pdf or doc.needs_pass:
                raise ValueError("Unreadable PDF")
            pages = doc.page_count
        if page > pages:
            raise HTTPException(422, "Physical page out of range.")
        snapshot.seek(0)
        result = snapshot, pages, target.name
        snapshot = None  # Caller now owns the verified, never-again-written snapshot.
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(404, "Original PDF unavailable.") from exc
    finally:
        if snapshot is not None:
            snapshot.close()


@pdf_router.post("/api/v1/document/pdf/resolve")
def resolve_pdf(request: PdfResolveRequest, _: None = Depends(require_bearer_token)) -> dict:
    kb_dir = _resolve_kb(request.kb)
    entries = HashRegistry(kb_dir / ".openkb/hashes.json").all_entries()
    matches = [
        (key, meta) for key, meta in entries.items() if meta.get("doc_name") == request.doc_name
    ]
    if len(matches) != 1:
        raise HTTPException(404, "Unique original PDF unavailable.")
    digest, meta = matches[0]
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise HTTPException(404, "Original PDF unavailable.")
    snapshot, pages, _filename = _pdf(kb_dir, meta, request.page, digest)
    snapshot.close()
    return {"hash": digest, "doc_name": request.doc_name, "page": request.page, "pages": pages}


@pdf_router.get("/api/v1/document/pdf")
def serve_pdf(
    kb: str,
    hash: str = Query(pattern=r"^[a-f0-9]{64}$"),
    page: str = Query(pattern=r"^[1-9][0-9]{0,8}$"),
    _: None = Depends(require_bearer_token),
) -> _SnapshotResponse:
    kb_dir = _resolve_kb(kb)
    meta = HashRegistry(kb_dir / ".openkb/hashes.json").get(hash)
    if meta is None:
        raise HTTPException(404, "Original PDF unavailable.")
    snapshot, _pages, filename = _pdf(kb_dir, meta, int(page), hash)
    try:
        return _SnapshotResponse(snapshot, filename, hash)
    except BaseException:
        snapshot.close()
        raise
