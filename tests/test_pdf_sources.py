"""Offline physical PDF provenance and serving regressions."""

import asyncio
import hashlib
import json

import pymupdf
import pytest
from fastapi.testclient import TestClient

from openkb.api import create_app


@pytest.fixture
def pdf_env(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    (kb / "raw").mkdir(parents=True)
    (kb / ".openkb").mkdir()
    (kb / "wiki").mkdir()
    pdf = kb / "raw" / "original.pdf"
    with pymupdf.open() as doc:
        for _ in range(13):
            doc.new_page()
        doc.save(pdf)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    meta = {"doc_name": "original", "raw_path": "raw/original.pdf", "type": "long_pdf"}
    registry = kb / ".openkb" / "hashes.json"
    registry.write_text(json.dumps({digest: meta}))
    monkeypatch.setenv("OPENKB_API_TOKEN", "test-secret")

    def resolve(name):
        if name != "test":
            raise ValueError("Unknown KB")
        return kb

    monkeypatch.setattr("openkb.api_helpers.resolve_kb_alias", resolve)
    return TestClient(create_app()), kb, digest, meta, registry


def test_resolve_and_serve_registered_pdf_at_physical_page(pdf_env):
    client, kb, digest, _, _ = pdf_env
    headers = {"Authorization": "Bearer test-secret"}
    response = client.post(
        "/api/v1/document/pdf/resolve",
        headers=headers,
        json={"kb": "test", "doc_name": "original", "page": 11},
    )
    assert response.status_code == 200
    assert response.json() == {"hash": digest, "doc_name": "original", "page": 11, "pages": 13}
    response = client.get(
        "/api/v1/document/pdf", headers=headers, params={"kb": "test", "hash": digest, "page": 11}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content == (kb / "raw/original.pdf").read_bytes()


@pytest.mark.parametrize("page", ["0", "-1", "14", "1.5", "1.0", "abc", "true"])
def test_invalid_physical_page_is_rejected(pdf_env, page):
    client, _, digest, _, _ = pdf_env
    response = client.get(
        "/api/v1/document/pdf",
        headers={"Authorization": "Bearer test-secret"},
        params={"kb": "test", "hash": digest, "page": page},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", ["resolve", "get"])
def test_changed_original_is_rejected_before_delivery(pdf_env, endpoint):
    client, kb, digest, _, _ = pdf_env
    original = kb / "raw/original.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(original)
    headers = {"Authorization": "Bearer test-secret"}
    if endpoint == "resolve":
        response = client.post(
            "/api/v1/document/pdf/resolve",
            headers=headers,
            json={"kb": "test", "doc_name": "original", "page": 1},
        )
    else:
        response = client.get(
            "/api/v1/document/pdf",
            headers=headers,
            params={"kb": "test", "hash": digest, "page": 1},
        )
    assert response.status_code == 404
    assert response.headers["content-type"] != "application/pdf"


@pytest.mark.parametrize("endpoint", ["resolve", "get"])
@pytest.mark.parametrize("change", ["replace", "overwrite"])
def test_original_change_after_hash_cannot_change_verified_snapshot(
    pdf_env, monkeypatch, endpoint, change
):
    client, kb, digest, _, _ = pdf_env
    original = kb / "raw/original.pdf"
    expected = original.read_bytes()
    replacement = kb / "raw/replacement.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(replacement)
    open_pdf = pymupdf.open

    def change_original(*args, **kwargs):
        # Deterministic ordinary file update between hashing and PDF parsing.
        if change == "replace":
            replacement.replace(original)
        else:
            original.write_bytes(replacement.read_bytes())
        return open_pdf(*args, **kwargs)

    monkeypatch.setattr("openkb.api_pdf_router.pymupdf.open", change_original)
    headers = {"Authorization": "Bearer test-secret"}
    if endpoint == "resolve":
        response = client.post(
            "/api/v1/document/pdf/resolve",
            headers=headers,
            json={"kb": "test", "doc_name": "original", "page": 11},
        )
        assert response.status_code == 200
        assert response.json()["pages"] == 13
    else:
        response = client.get(
            "/api/v1/document/pdf",
            headers=headers,
            params={"kb": "test", "hash": digest, "page": 11},
        )
        assert response.status_code == 200
        assert response.content == expected
        assert hashlib.sha256(response.content).hexdigest() == digest


def test_byte_ranges_supported(pdf_env):
    client, kb, digest, _, _ = pdf_env
    response = client.get(
        "/api/v1/document/pdf",
        params={"kb": "test", "hash": digest, "page": 13},
        headers={"Authorization": "Bearer test-secret", "Range": "bytes=0-4"},
    )
    assert response.status_code == 206
    assert response.content == b"%PDF-"
    assert response.headers["content-type"] == "application/pdf"


@pytest.mark.parametrize(
    "stored", ["../outside.pdf", "/etc/passwd", "wiki/secret.pdf", "raw/link.pdf"]
)
def test_registry_paths_cannot_escape_raw_or_kb(pdf_env, stored):
    client, kb, digest, meta, registry = pdf_env
    outside = kb.parent / "outside.pdf"
    outside.write_bytes((kb / "raw/original.pdf").read_bytes())
    (kb / "raw/link.pdf").symlink_to(outside)
    (kb / "wiki/secret.pdf").write_bytes(outside.read_bytes())
    meta["raw_path"] = stored
    registry.write_text(json.dumps({digest: meta}))
    headers = {"Authorization": "Bearer test-secret"}
    assert (
        client.get(
            "/api/v1/document/pdf",
            headers=headers,
            params={"kb": "test", "hash": digest, "page": 1},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/document/pdf/resolve",
            headers=headers,
            json={"kb": "test", "doc_name": "original", "page": 1},
        ).status_code
        == 404
    )


def test_unregistered_pdf_cannot_be_selected_by_path_or_hash(pdf_env):
    client, kb, digest, _, registry = pdf_env
    registry.write_text("{}")
    response = client.get(
        "/api/v1/document/pdf",
        headers={"Authorization": "Bearer test-secret"},
        params={"kb": "test", "hash": digest, "page": 1, "path": "raw/original.pdf"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("token", [None, "wrong"])
def test_both_pdf_endpoints_enforce_bearer_auth(pdf_env, token):
    client, _, digest, _, _ = pdf_env
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    assert (
        client.get(
            "/api/v1/document/pdf",
            headers=headers,
            params={"kb": "test", "hash": digest, "page": 1},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/document/pdf/resolve",
            headers=headers,
            json={"kb": "test", "doc_name": "original", "page": 1},
        ).status_code
        == 401
    )


@pytest.mark.parametrize("kind", ["missing", "fake", "cloud", "duplicate"])
def test_unavailable_original_has_no_resolvable_link(pdf_env, kind):
    client, kb, digest, meta, registry = pdf_env
    if kind == "missing":
        (kb / "raw/original.pdf").unlink()
    elif kind == "fake":
        (kb / "raw/original.pdf").write_text("not a PDF")
    elif kind == "cloud":
        meta.pop("raw_path")
        meta["path"] = "pageindex-cloud:remote"
    entries = {digest: meta}
    if kind == "duplicate":
        entries["a" * 64] = dict(meta)
    registry.write_text(json.dumps(entries))
    response = client.post(
        "/api/v1/document/pdf/resolve",
        headers={"Authorization": "Bearer test-secret"},
        json={"kb": "test", "doc_name": "original", "page": 1},
    )
    assert response.status_code == 404


def test_symlink_loop_original_is_unavailable(pdf_env):
    client, kb, digest, meta, registry = pdf_env
    (kb / "raw/loop.pdf").symlink_to("loop.pdf")
    meta["raw_path"] = "raw/loop.pdf"
    registry.write_text(json.dumps({digest: meta}))
    response = client.post(
        "/api/v1/document/pdf/resolve",
        headers={"Authorization": "Bearer test-secret"},
        json={"kb": "test", "doc_name": "original", "page": 11},
    )
    assert response.status_code == 404


def test_resolver_does_not_advertise_malformed_registry_hash(pdf_env):
    client, _, _, meta, registry = pdf_env
    registry.write_text(json.dumps({"not-a-sha256": meta}))
    response = client.post(
        "/api/v1/document/pdf/resolve",
        headers={"Authorization": "Bearer test-secret"},
        json={"kb": "test", "doc_name": "original", "page": 11},
    )
    assert response.status_code == 404


def test_wrong_kb_cannot_resolve_registered_original(pdf_env, monkeypatch):
    client, kb, digest, _, _ = pdf_env
    other = kb.parent / "other"
    (other / ".openkb").mkdir(parents=True)
    (other / "wiki").mkdir()
    monkeypatch.setattr("openkb.api_helpers.resolve_kb_alias", lambda name: other)
    assert (
        client.get(
            "/api/v1/document/pdf",
            headers={"Authorization": "Bearer test-secret"},
            params={"kb": "other", "hash": digest, "page": 11},
        ).status_code
        == 404
    )


@pytest.mark.parametrize("page", [None, True, 1.0, "11", 0, 14])
def test_resolver_requires_valid_integer_page(pdf_env, page):
    client, *_ = pdf_env
    assert (
        client.post(
            "/api/v1/document/pdf/resolve",
            headers={"Authorization": "Bearer test-secret"},
            json={"kb": "test", "doc_name": "original", "page": page},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "range_header, expected_status",
    [
        (None, 200),
        ("bytes=0-4", 206),
        ("bytes=-5", 206),
        ("bytes=5-", 206),
        ("bytes=0-4,10-14", 206),
        ("bytes=999999-", 416),
        ("items=0-4", 400),
    ],
)
def test_delivery_uses_owned_handle_even_after_original_replacement(
    pdf_env, range_header, expected_status
):
    from openkb.api_pdf_router import serve_pdf

    _, kb, digest, _, _ = pdf_env
    original = kb / "raw/original.pdf"
    expected = original.read_bytes()
    response = serve_pdf("test", digest, "11")
    snapshot = response.snapshot
    messages = []

    async def send(message):
        if message["type"] == "http.response.start":
            replacement = kb / "raw/new.pdf"
            replacement.write_bytes(b"ordinary replacement after validation")
            replacement.replace(original)
        messages.append(message)

    async def receive():
        raise AssertionError("ASGI 2.4 does not require disconnect polling")

    scope = {
        "type": "http",
        "method": "GET",
        "asgi": {"spec_version": "2.4"},
        "extensions": {"http.response.pathsend": {}},
        "headers": [(b"range", range_header.encode())] if range_header else [],
    }
    asyncio.run(response(scope, receive, send))
    assert snapshot.closed
    assert messages[0]["status"] == expected_status
    assert all(m["type"] != "http.response.pathsend" for m in messages)
    content = b"".join(m.get("body", b"") for m in messages)
    if range_header is None:
        assert content == expected
    elif range_header == "bytes=0-4":
        assert content == expected[:5]
    elif range_header == "bytes=-5":
        assert content == expected[-5:]
    elif range_header == "bytes=5-":
        assert content == expected[5:]
    elif range_header == "bytes=0-4,10-14":
        assert expected[:5] in content
        assert expected[10:15] in content
        assert b"Content-Range: bytes 0-4/" in content
        assert b"Content-Range: bytes 10-14/" in content


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
def test_delivery_failure_closes_snapshot(pdf_env, failure):
    from openkb.api_pdf_router import serve_pdf

    _, _, digest, _, _ = pdf_env
    response = serve_pdf("test", digest, "11")

    async def send(message):
        if message["type"] == "http.response.body":
            raise failure("interrupted transfer")

    async def receive():
        raise AssertionError("unexpected polling")

    with pytest.raises(failure):
        asyncio.run(
            response(
                {
                    "type": "http",
                    "method": "GET",
                    "asgi": {"spec_version": "2.4"},
                    "headers": [],
                },
                receive,
                send,
            )
        )
    assert response.snapshot.closed


@pytest.mark.parametrize("case", ["resolve", "get", "page", "digest", "parse"])
def test_snapshot_closed_after_validation_and_requests(pdf_env, monkeypatch, case):
    import openkb.api_pdf_router as router

    client, kb, digest, meta, registry = pdf_env
    handles = []
    make_temp = router.tempfile.TemporaryFile

    def track_temp(*args, **kwargs):
        handle = make_temp(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(router.tempfile, "TemporaryFile", track_temp)
    if case == "digest":
        (kb / "raw/original.pdf").write_bytes(b"changed original")
    elif case == "parse":
        invalid = b"%PDF-not a readable document"
        (kb / "raw/original.pdf").write_bytes(invalid)
        digest = hashlib.sha256(invalid).hexdigest()
        registry.write_text(json.dumps({digest: meta}))
    headers = {"Authorization": "Bearer test-secret"}
    if case == "resolve":
        result = client.post(
            "/api/v1/document/pdf/resolve",
            headers=headers,
            json={"kb": "test", "doc_name": "original", "page": 11},
        )
    else:
        result = client.get(
            "/api/v1/document/pdf",
            headers=headers,
            params={"kb": "test", "hash": digest, "page": 14 if case == "page" else 11},
        )
    assert (
        result.status_code
        == {
            "resolve": 200,
            "get": 200,
            "page": 422,
            "digest": 404,
            "parse": 404,
        }[case]
    )
    assert handles
    assert all(handle.closed for handle in handles)


def test_snapshot_copy_failure_closes_temporary_file(pdf_env, monkeypatch):
    import openkb.api_pdf_router as router

    client, _, digest, _, _ = pdf_env
    make_temp = router.tempfile.TemporaryFile
    handles = []

    class FailedDisk:
        def __init__(self):
            self.handle = make_temp(mode="w+b")
            handles.append(self.handle)

        def write(self, chunk):
            raise OSError("temporary disk full")

        def close(self):
            self.handle.close()

    monkeypatch.setattr(router.tempfile, "TemporaryFile", lambda **kwargs: FailedDisk())
    result = client.get(
        "/api/v1/document/pdf",
        headers={"Authorization": "Bearer test-secret"},
        params={"kb": "test", "hash": digest, "page": 11},
    )
    assert result.status_code == 404
    assert handles and all(handle.closed for handle in handles)


def test_response_initialization_error_closes_verified_snapshot(pdf_env, monkeypatch):
    import openkb.api_pdf_router as router

    _, _, digest, _, _ = pdf_env
    handles = []

    def fail_response(snapshot, filename, digest):
        handles.append(snapshot)
        raise OSError("response initialization failed")

    monkeypatch.setattr(router, "_SnapshotResponse", fail_response)
    with pytest.raises(OSError, match="response initialization failed"):
        router.serve_pdf("test", digest, "11")
    assert handles and all(handle.closed for handle in handles)


@pytest.mark.parametrize("if_range", ["matching", "stale"])
def test_snapshot_preserves_if_range_semantics(pdf_env, if_range):
    client, kb, digest, _, _ = pdf_env
    headers = {"Authorization": "Bearer test-secret"}
    params = {"kb": "test", "hash": digest, "page": 11}
    full = client.get("/api/v1/document/pdf", params=params, headers=headers)
    from openkb.api_pdf_router import serve_pdf

    response = serve_pdf("test", digest, "11")
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        raise AssertionError("unexpected polling")

    validator = full.headers["etag"] if if_range == "matching" else '"stale"'
    asyncio.run(
        response(
            {
                "type": "http",
                "method": "GET",
                "asgi": {"spec_version": "2.4"},
                "headers": [(b"range", b"bytes=0-4"), (b"if-range", validator.encode())],
            },
            receive,
            send,
        )
    )
    assert messages[0]["status"] == (206 if if_range == "matching" else 200)
    content = b"".join(m.get("body", b"") for m in messages)
    assert content == (b"%PDF-" if if_range == "matching" else full.content)
    assert response.snapshot.closed
