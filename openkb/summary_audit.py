"""Exact PageIndex render ownership for oversized summary audits, never evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openkb import frontmatter
from openkb.tree_renderer import _render_nodes_summary


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class Fragment:
    key: str
    start: int
    end: int
    owned: str
    context: str


@dataclass(frozen=True)
class SummaryArtifact:
    parent: str
    text: str
    preamble: str
    sections: tuple[str, ...]
    fragments: tuple[Fragment, ...]
    source_snapshot: str
    tree_snapshot: str
    tree: dict
    manifest_digest: str
    target_digest: str
    source_digest: str
    tree_digest: str
    guards: tuple[tuple[Path, str], ...] = ()
    tree_reader: Callable[[], dict] | None = None

    def verify(self, pages, candidates):
        from openkb.grounding import GroundingError

        try:
            stored_changed = any(
                hashlib.sha256(p.read_bytes()).hexdigest() != sha for p, sha in self.guards
            ) or (
                self.tree_reader is not None and snapshot(self.tree_reader()) != self.tree_snapshot
            )
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
            raise GroundingError("Grounding frozen summary inputs unavailable") from exc
        if (
            candidates.get(self.parent) != self.text
            or digest(self.text) != self.target_digest
            or digest(self.source_snapshot) != self.source_digest
            or digest(self.tree_snapshot) != self.tree_digest
            or snapshot(pages) != self.source_snapshot
            or snapshot(self.tree) != self.tree_snapshot
            or digest(
                snapshot([(f.key, f.start, f.end, f.owned, f.context) for f in self.fragments])
            )
            != self.manifest_digest
            or stored_changed
        ):
            raise GroundingError("Grounding frozen summary inputs changed; manual review required")
        cursor = 0
        keys = set()
        for f in self.fragments:
            expected = f"{self.parent}#span:{f.start}:{f.end}:{digest(f.owned)}"
            if (
                f.start != cursor
                or f.end <= f.start
                or f.key != expected
                or f.key in keys
                or self.text[f.start : f.end] != f.owned
            ):
                raise GroundingError("Grounding summary fragment coverage invalid")
            cursor = f.end
            keys.add(f.key)
        if cursor != len(self.text) or "".join(f.owned for f in self.fragments) != self.text:
            raise GroundingError("Grounding summary fragment coverage incomplete")

    def plan(self, pages, *, doc_name=None):
        from openkb.grounding import audit_messages

        self.verify(pages, {self.parent: self.text})
        return [
            audit_messages(pages, {f.key: f.context + f.owned}, doc_name=doc_name)
            for f in self.fragments
        ]


def freeze_summary(
    parent, text, pages, tree, *, prefix="", guards=(), tree_reader=None
) -> SummaryArtifact:
    """Admit only an exact unannotated renderer output and unambiguous node map."""
    from openkb.grounding import GroundingError

    try:
        if not parent.startswith("summaries/") or not text.startswith(prefix):
            raise ValueError
        raw = text[len(prefix) :]
        parts = frontmatter.split(raw)
        meta = frontmatter.parse(raw)
        if (
            parts is None
            or meta.get("doc_type") != "pageindex"
            or meta.get("type") != "Summary"
            or meta.get("full_text") != f"sources/{parent.removeprefix('summaries/')}.json"
            or not parts[1].startswith("\n")
        ):
            raise ValueError
        preamble = prefix + parts[0] + "\n"
        flat: list[tuple[dict, tuple[int, ...]]] = []
        ids = set()

        def walk(nodes, ancestors=()):
            if not isinstance(nodes, list):
                raise ValueError
            for node in nodes:
                node_id = node["node_id"]
                start, end = node["start_index"], node["end_index"]
                if (
                    not isinstance(node_id, str)
                    or not node_id
                    or node_id in ids
                    or type(start) is not int
                    or type(end) is not int
                    or not 1 <= start <= end
                    or not isinstance(node["title"], str)
                    or "\n" in node["title"]
                    or not isinstance(node.get("summary", ""), str)
                ):
                    raise ValueError
                for a in ancestors:
                    n = flat[a][0]
                    if not n["start_index"] <= start <= end <= n["end_index"]:
                        raise ValueError
                ids.add(node_id)
                i = len(flat)
                flat.append((node, ancestors))
                walk(node.get("nodes", []), ancestors + (i,))

        walk(tree["structure"])
        rendered = _render_nodes_summary(tree["structure"], 1)
        if text != preamble + rendered:
            raise ValueError
        matches = list(re.finditer(r"(?m)^(#{1,6}) (.+) \(pages (\d+)–(\d+)\)\n", text))
        if not flat or len(matches) != len(flat) or matches[0].start() != len(preamble):
            raise ValueError
        sections = []
        for i, ((node, ancestors), match) in enumerate(zip(flat, matches, strict=True)):
            if (
                match[1] != "#" * min(len(ancestors) + 1, 6)
                or match[2] != node["title"]
                or (int(match[3]), int(match[4])) != (node["start_index"], node["end_index"])
            ):
                raise ValueError
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            own = text[match.start() : end]
            if own[len(match[0]) :].strip("\n") != (
                "Summary: " + node["summary"] if node.get("summary") else ""
            ):
                raise ValueError
            sections.append(own)
        fragments = []
        for i, (_, ancestors) in enumerate(flat):
            start = matches[i].start() if i else 0
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            own = text[start:end]
            context = preamble + "".join(sections[a] for a in ancestors) if i else ""
            key = f"{parent}#span:{start}:{end}:{digest(own)}"
            fragments.append(Fragment(key, start, end, own, context))
        artifact = SummaryArtifact(
            parent,
            text,
            preamble,
            tuple(sections),
            tuple(fragments),
            snapshot(pages),
            snapshot(tree),
            tree,
            digest(snapshot([(f.key, f.start, f.end, f.owned, f.context) for f in fragments])),
            digest(text),
            digest(snapshot(pages)),
            digest(snapshot(tree)),
            tuple(guards),
            tree_reader,
        )
        artifact.verify(pages, {parent: text})
        return artifact
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        raise GroundingError(
            "Grounding summary/tree mapping ambiguous or stale; manual review required"
        ) from exc


def load_summary_artifact(kb_dir, summary_path, doc_id, doc_name, pages, *, prefix):
    """Read stored tree without constructing a client, indexing, or DB writes."""
    from openkb.grounding import GroundingError, load_source_pages

    db = kb_dir / ".openkb/pageindex.db"
    source = kb_dir / f"wiki/sources/{doc_name}.json"

    def read_tree():
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT doc_name, doc_description, structure FROM documents "
                "WHERE doc_id = ? AND collection_name = ?",
                (doc_id, "default"),
            ).fetchall()
        if len(rows) != 1 or Path(rows[0][0]).stem != doc_name:
            raise GroundingError("Grounding stored summary tree identity mismatch")
        name, description, structure = rows[0]
        return {
            "doc_name": name,
            "doc_description": description,
            "structure": json.loads(structure),
            "stored_structure": structure,
        }

    try:
        guards = tuple(
            (p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in (summary_path, source)
        )
        tree = read_tree()
        if load_source_pages(kb_dir / "wiki", doc_name) != pages:
            raise GroundingError("Grounding frozen source inputs changed")
        artifact = freeze_summary(
            f"summaries/{doc_name}",
            prefix + summary_path.read_text(encoding="utf-8"),
            pages,
            tree,
            prefix=prefix,
            guards=guards,
            tree_reader=read_tree,
        )
        return artifact
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        raise GroundingError(
            "Grounding stored summary tree unavailable; manual review required"
        ) from exc
