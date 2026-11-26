"""Transactional recompilation of existing wiki pages, never the index cache."""

from contextlib import contextmanager
from pathlib import Path

from openkb.add_coordinator import DirtyRollbackError
from openkb.locks import kb_ingest_lock_held
from openkb.mutation import snapshot_paths


@contextmanager
def recompile_mutation(kb_dir: Path, doc_name: str):
    """Restore prior pages on any failure; callers must stop on dirty rollback.

    Compiler writes are atomic, so hardlinked page backups are safe. Shared
    concept/entity pages must be included, but raw files, PageIndex databases,
    and completed index caches are deliberately outside this transaction.
    """
    if not kb_ingest_lock_held(kb_dir / ".openkb"):
        raise RuntimeError("recompile_mutation requires kb_ingest_lock")
    wiki = kb_dir / "wiki"
    page_dirs = {wiki / "concepts", wiki / "entities"}
    snapshot = snapshot_paths(
        kb_dir,
        [wiki / "summaries" / f"{doc_name}.md", wiki / "index.md", *sorted(page_dirs)],
        operation="recompile",
        details={"doc_name": doc_name},
        hardlink_dirs=page_dirs,
    )
    try:
        yield
        snapshot.mark_committed()
    except BaseException as exc:
        rollback_error = snapshot.rollback_best_effort()
        if rollback_error is None:
            snapshot.discard_best_effort()
        elif isinstance(exc, Exception):
            raise DirtyRollbackError("recompile", snapshot.journal_path) from exc
        raise
    else:
        snapshot.discard_best_effort()
