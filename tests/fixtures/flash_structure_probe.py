"""Subprocess fixture: actual Flash SDK with network/auth/inference blocked."""

import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["HOME"] = str(Path(sys.argv[2]).parent)


def blocked(*args, **kwargs):
    raise AssertionError("offline: network/auth/inference forbidden")


socket.socket.connect = blocked


# Guard provider credentials before importing SDK; no auth/config discovery allowed.
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        if any(
            piece in name
            for piece in (
                "/.claude/",
                "/.codex/",
                "/.config/litellm/",
                "credentials.json",
                "auth.json",
            )
        ):
            blocked()


sys.addaudithook(audit)
import litellm  # noqa: E402 - fail closed before SDK/provider imports

litellm.completion = litellm.acompletion = blocked
from pageindex.flash import api, main  # noqa: E402

spec = importlib.util.spec_from_file_location("structure", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if len(sys.argv) > 3 and sys.argv[3] == "--malformed":
    api_extract, process = api.extract_toc, main.process_page
    try:
        with module.flash_structure_policy():
            api.page_index_flash(sys.argv[2], summary=False, optimize=False)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("malformed PDF was accepted")
    assert api.extract_toc is api_extract and main.process_page is process
    print(json.dumps({"rejected": True, "restored": True}))
    sys.exit(0)

original = main.parse_charlevel_meta_parallel
main.parse_charlevel_meta_parallel = lambda pdf, **kw: original(pdf, workers=1)
before = main.extract_toc(sys.argv[2], workers=1)
api_extract = api.extract_toc
process = main.process_page
captured = []
# Observe the seam's return, proving page text identity is retained before API pops it.
with module.flash_structure_policy():
    repaired_extract = api.extract_toc

    def observe(*args, **kwargs):
        r = repaired_extract(*args, **kwargs)
        captured.append(r["page_texts"])
        return r

    api.extract_toc = observe
    after = api.page_index_flash(sys.argv[2], summary=False, optimize=False)
assert api.extract_toc is api_extract
assert main.process_page is process
assert captured[0] == before["page_texts"]


def walk(nodes):
    for n in nodes:
        yield n
        yield from walk(n.get("nodes", []))


nodes = list(walk(after["structure"]))
assert len({n["node_id"] for n in nodes}) == len(nodes)
for n in nodes:
    assert 1 <= n["start_index"] <= n["end_index"] <= len(captured[0])
    for child in n.get("nodes", []):
        assert n["start_index"] <= child["start_index"] <= child["end_index"] <= n["end_index"]
assert not any("summary" in n or "text" in n for n in nodes)
from pageindex.utils import leaf_own_text  # noqa: E402

own_text = {
    n["title"]: leaf_own_text(after["structure"], n, [(p, 0) for p in captured[0]])
    for n in nodes
    if not n.get("nodes")
}
print(
    json.dumps(
        {
            "before": before["structure"],
            "after": after["structure"],
            "pages": len(captured[0]),
            "raw_pages_equal": True,
            "leaf_text": own_text,
        }
    )
)
