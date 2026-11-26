"""Conservative, LLM-free repair of Flash 0.7.4 layout before summaries.

This module never writes PDFs, pages, images, summaries, or source excerpts.
``repair_structure`` also accepts normalized layout dictionaries for offline tests.
The optional worker-local context is the only pinned private-SDK seam.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version
from threading import RLock

_SEAM_LOCK = RLock()


@contextmanager
def flash_structure_policy():
    """Worker-local SDK 0.7.4 seam, applied before intro nodes and summaries.

    Use around ``client.submit_document(..., mode='flash')`` alongside the
    existing completion quota policy. No completion function is replaced here.
    Public Flash parameters cannot recover headings lost during block joining.
    Always restore SDK globals, including on parser errors; serialize this seam.
    """
    if version("pageindex") != "0.7.4":
        raise ValueError("Unsupported structural repair SDK")
    from pageindex.flash import api, main

    with _SEAM_LOCK:
        original_extract = api.extract_toc
        original_process = main.process_page

        def extract(pdf, **kwargs):
            captured = []

            def process(*args, **kw):
                page = original_process(*args, **kw)
                captured.append(page)
                return page

            main.process_page = process
            try:
                # Public low-level control: avoid unbounded auto CPU fan-out.
                kwargs["workers"] = 1
                result = original_extract(pdf, **kwargs)
            finally:
                main.process_page = original_process
            return repair_structure(result, _normalize_pages(captured))

        api.extract_toc = extract
        try:
            yield
        finally:
            api.extract_toc = original_extract
            main.process_page = original_process


def _normalize_pages(pages):
    normalized = []
    for page in pages:
        excluded = {
            id(line)
            for block in page.blocks
            if block.type in (1, 2, 12)
            for line in block.primary_slot
        }
        lines = [
            dict(
                text="".join(s.text for s in line.primary_slot).strip(),
                top=line.top_edge(),
                bottom=line.bottom_edge(),
                left=line.left_edge(),
                size=line.avg_font_size(),
                bold=line.bold_frac() > 0.9,
                excluded=id(line) in excluded,
            )
            for line in page.lines
        ]
        normalized.append(
            dict(number=page.page_index, height=page.bounds.bbox_height(), lines=lines)
        )
    return normalized


def _key(text):
    return "".join(c.casefold() for c in text if c.isalnum())


def _walk(nodes):
    for node in nodes:
        yield node
        yield from _walk(node.get("nodes", []))


def _position(page, line):
    return page["number"], -line["top"]


def repair_structure(result, pages):
    """Return a copy of the semantic tree; keep all raw result fields untouched.

    Only recover isolated, unpunctuated short lines with heading-like spacing,
    followed by flowing body, under an existing larger-font section. This is
    deliberately not a general document outline guesser.
    """
    repaired = dict(result)
    repaired["structure"] = copy.deepcopy(result.get("structure", []))
    _filter_noise(repaired["structure"], pages)
    flat = list(_walk(repaired["structure"]))
    if not flat:
        return repaired
    layouts = [(page, line) for page in pages for line in page["lines"]]
    locations = {}
    for node in flat:
        found = next(
            (
                (page, line)
                for page, line in layouts
                if page["number"] == node["start_index"]
                and _key(line["text"]) == _key(node["title"])
            ),
            None,
        )
        if found:
            locations[id(node)] = found
    sizes = Counter()
    for _, line in layouts:
        if not line.get("excluded") and len(line["text"]) > 60:
            sizes[round(line["size"], 1)] += len(line["text"])
    if not sizes:
        return repaired
    body_size = sizes.most_common(1)[0][0]
    used = {str(n.get("node_id")) for n in flat}
    additions = {}
    for page in pages:
        lines = [line for line in page["lines"] if not line.get("excluded")]
        for index, heading in enumerate(lines):
            title = heading["text"].strip()
            if not (1 <= index < len(lines) - 1):
                continue
            above, below = lines[index - 1], lines[index + 1]
            if not (
                2 <= len(title.split()) <= 10
                and len(title) <= 90
                and title[0].isupper()
                and not re.search(r"[.!?:;,\u2022]|^[\d\W]", title)
                and abs(heading["size"] - body_size) <= 0.6
                and abs(below["size"] - body_size) <= 0.6
                and len(below["text"]) > 60
                and abs(heading["left"] - below["left"]) <= 2
                and above["bottom"] - heading["top"] >= 1.7 * body_size
                and 0.6 * body_size <= heading["bottom"] - below["top"] <= 1.4 * body_size
            ):
                continue
            position = _position(page, heading)
            owners = []
            for order, node in enumerate(flat):
                location = locations.get(id(node))
                if location is None or location[1]["size"] < heading["size"] + 0.5:
                    continue
                if not node["start_index"] <= page["number"] <= node["end_index"]:
                    continue
                start = _position(*location)
                # A shared boundary page belongs to the next section only below
                # its actual heading, never merely because the page number ties.
                descendants = {id(n) for n in _walk(node.get("nodes", []))}
                next_start = next(
                    (
                        _position(*locations[id(n)])
                        for n in flat[order + 1 :]
                        if id(n) in locations
                        and id(n) not in descendants
                        and n["start_index"] <= node["end_index"]
                        and n["start_index"] >= node["start_index"]
                    ),
                    None,
                )
                if start < position and (next_start is None or position < next_start):
                    owners.append((node["end_index"] - node["start_index"], -order, node))
            if not owners or any(
                _key(n["title"]) == _key(title) and n["start_index"] == page["number"] for n in flat
            ):
                continue
            owner = min(owners, key=lambda entry: entry[:2])[2]
            node_id = "layout-" + str(page["number"]) + "-" + str(index)
            while node_id in used:
                node_id += "-new"
            used.add(node_id)
            child = dict(
                title=title,
                node_id=node_id,
                start_index=page["number"],
                end_index=owner["end_index"],
            )
            additions.setdefault(id(owner), []).append((position, child))
    for owner in flat:
        entries = list(additions.get(id(owner), []))
        if not entries:
            continue
        entries.extend(
            (_position(*locations[id(n)]), n) for n in owner.get("nodes", []) if id(n) in locations
        )
        # Never replace an existing child whose layout position is unknown.
        if len(entries) != len(additions[id(owner)]) + len(owner.get("nodes", [])):
            continue
        entries.sort(key=lambda item: item[0])
        for index, (_, node) in enumerate(entries[:-1]):
            node["end_index"] = max(node["start_index"], entries[index + 1][1]["start_index"])
        owner["nodes"] = [node for _, node in entries]
    return repaired


def _filter_noise(nodes, pages):
    """Drop only leaf CTAs at the lower page edge with no flowing body below."""
    page_map = {p["number"]: p for p in pages}
    kept = []
    for node in nodes:
        if node.get("nodes"):
            _filter_noise(node["nodes"], pages)
            kept.append(node)
            continue
        page = page_map.get(node["start_index"])
        match = (
            None
            if page is None
            else next(
                (line for line in page["lines"] if _key(line["text"]) == _key(node["title"])), None
            )
        )
        cta = re.match(r"^(get|subscribe|sign\s+up|join|follow)\b", node["title"], re.I)
        target = re.search(
            r"\b(newsletter|updates|mailing\s+list|community|emails?)\b", node["title"], re.I
        )
        edge = match is not None and match["top"] < 0.12 * page["height"]
        body_after = match is not None and any(
            not line.get("excluded") and line["top"] < match["bottom"] and len(line["text"]) > 60
            for line in page["lines"]
        )
        if not (
            cta and target and edge and not body_after and node["start_index"] == node["end_index"]
        ):
            kept.append(node)
    nodes[:] = kept
