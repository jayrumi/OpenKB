"""Offline structural regressions independent of the app's legacy PageIndex."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from openkb.flash_structure import repair_structure

FLASH_PYTHON = Path("/home/jayrumi/openkb-development/flash-venv/bin/python")
REAL_PDF = Path(
    "/home/jayrumi/openkb-development/ab-anthropic-source-control/input/anthropic-source-control.pdf"
)


def run_sdk(pdf, *extra):
    if not FLASH_PYTHON.exists():
        pytest.skip("separately pinned Flash SDK environment is not installed")
    probe = Path(__file__).parent / "fixtures" / "flash_structure_probe.py"
    module = Path(__file__).parents[1] / "openkb" / "flash_structure.py"
    run = subprocess.run(
        [str(FLASH_PYTHON), str(probe), str(module), str(pdf), *extra],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert run.returncode == 0, run.stderr + run.stdout
    return json.loads(run.stdout)


def test_real_local_flash_parser_preserves_technical_subsections():
    if not REAL_PDF.exists():
        pytest.skip("external read-only 13-page acceptance PDF is not installed")
    output = run_sdk(REAL_PDF)
    assert output["pages"] == 13
    before = {n["title"] for n in output["before"]}
    assert "Data structures" not in before
    root = output["after"]
    owner = next(n for n in root if n["title"] == "How Scion works")
    assert [n["title"] for n in owner["nodes"]] == ["Data structures", "Scaling Scion"]
    assert [(n["start_index"], n["end_index"]) for n in owner["nodes"]] == [(5, 7), (7, 8)]
    assert "Get the developer newsletter" not in {n["title"] for n in root}
    assert "Scaling Scion" not in output["leaf_text"]["Data structures"]
    assert "What we learned building Scion" not in output["leaf_text"]["Scaling Scion"]
    assert "unit of work" in output["leaf_text"]["Data structures"]
    assert "small fleet" in output["leaf_text"]["Scaling Scion"]


def line(text, top, bottom, size=12.75, left=51.75, bold=False):
    return dict(text=text, top=top, bottom=bottom, size=size, left=left, bold=bold, excluded=False)


def example():
    # Actual SDK 0.7.4 geometry: Type3 font names and bold flags are lost;
    # both technical headings are joined into body blocks, not candidates.
    pages = [dict(number=i, height=841.92, lines=[]) for i in range(1, 14)]
    pages[4]["lines"] = [
        line("How Scion works", 613.17, 598.17, 15),
        line("Scion is a source control system with a novel storage architecture.", 586.92, 574.17),
        line("It implements a write-ahead log and serves data to many clients.", 547.17, 534.42),
        line("Data structures", 505.92, 493.17),
        line(
            "In Scion, the unit of work is a changeset, an entry in a write-ahead log.",
            481.92,
            469.17,
        ),
    ]
    pages[6]["lines"] = [
        line("A paragraph continued from the previous page describes its storage.", 785.67, 772.92),
        line("Another complete line introduces the distributed architecture.", 772.92, 760.17),
        line("Scaling Scion", 731.67, 718.92),
        line(
            "A small fleet of Scion servers provides source control APIs for clients.",
            708.42,
            695.67,
        ),
    ]
    result = dict(
        doc_title="Redesigning source control at Anthropic",
        page_texts=["raw page"] * 13,
        structure=[
            dict(
                title="How Scion works",
                node_id="0002",
                start_index=5,
                end_index=8,
                custom="preserve",
            ),
            dict(title="What we learned", node_id="0003", start_index=8, end_index=13),
        ],
        images=[{"bytes": "unchanged"}],
    )
    return result, pages


def test_actual_merged_body_size_headings_are_recovered_before_summaries():
    source, pages = example()
    before = deepcopy(source)
    repaired = repair_structure(source, pages)

    def walk(nodes):
        for n in nodes:
            yield n
            yield from walk(n.get("nodes", []))

    nodes = {n["title"]: n for n in walk(repaired["structure"])}
    assert "Data structures" in nodes, "SDK swallowed a same-size heading into body"
    assert [n["title"] for n in nodes["How Scion works"]["nodes"]] == [
        "Data structures",
        "Scaling Scion",
    ]
    assert (nodes["Data structures"]["start_index"], nodes["Data structures"]["end_index"]) == (
        5,
        7,
    )
    assert (nodes["Scaling Scion"]["start_index"], nodes["Scaling Scion"]["end_index"]) == (7, 8)
    assert nodes["How Scion works"]["custom"] == "preserve"
    assert all("summary" not in n and "text" not in n for n in walk(repaired["structure"]))
    assert source == before
    assert repaired["page_texts"] is source["page_texts"]
    assert repaired["images"] is source["images"]


def test_only_high_confidence_edge_cta_noise_is_removed():
    source, pages = example()
    titles = [
        "Newsletter architecture",
        "Acknowledgements",
        "Footnotes",
        "Get the developer newsletter",
    ]
    for i, title in enumerate(titles):
        source["structure"].append(
            dict(title=title, node_id=f"tail-{i}", start_index=13, end_index=13)
        )
    pages[12]["lines"] = [
        line("Newsletter architecture", 500, 485, 15),
        line("This technical section explains how email queues preserve event ordering.", 470, 457),
        line("Acknowledgements", 331.92, 316.92, 15),
        line("Footnotes", 205.17, 190.17, 15),
        line("Get the developer newsletter", 67.92, 52.92, 15, left=75.375),
    ]
    repaired = repair_structure(source, pages)
    names = [n["title"] for n in repaired["structure"]]
    assert "Get the developer newsletter" not in names
    assert all(t in names for t in titles[:-1])
    assert repaired["page_texts"] is source["page_texts"]


def test_midpage_cta_title_with_substantive_body_is_not_noise():
    source, pages = example()
    source["structure"].append(
        dict(title="Subscribe to updates", start_index=13, end_index=13, node_id="real")
    )
    pages[12]["lines"] = [
        line("Subscribe to updates", 430, 415, 15),
        line(
            "This API subscribes to durable updates emitted by the distributed transaction log.",
            400,
            387,
        ),
    ]
    assert repair_structure(source, pages)["structure"][-1]["node_id"] == "real"


def test_generated_multipage_pdf_without_toc_keeps_figures(tmp_path):
    import pymupdf

    pdf = tmp_path / "synthetic.pdf"
    doc = pymupdf.open()
    for index, title in enumerate(("Engine overview", "Storage mechanisms", "Integration notes")):
        page = doc.new_page(width=595, height=842)
        if index == 0:
            page.insert_text((52, 55), "Distributed engine design", fontsize=24)
        page.insert_text((52, 100), title, fontsize=16)
        for row in range(6):
            page.insert_text(
                (52, 128 + row * 22),
                f"Section {index + 1}: this paragraph describes consistent transaction "
                "handling for many clients.",
                fontsize=12,
            )
        if index == 0:
            page.insert_text((52, 278.5), "Transaction layout", fontsize=12)
            page.insert_text(
                (52, 301.75),
                "Each transaction records its dependencies and preserves ordering "
                "across storage replicas.",
                fontsize=12,
            )
            page.insert_text(
                (52, 323.75),
                "This following body line contains the additional implementation "
                "details required here.",
                fontsize=12,
            )
            page.insert_image(
                pymupdf.Rect(52, 350, 152, 380),
                stream=bytes.fromhex(
                    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000b49444154789c636000020000050001a5f645400000000049454e44ae426082"
                ),
            )
        for row in range(12):
            page.insert_text(
                (52, 400 + row * 20),
                f"Page {index + 1} row {row}: clients preserve consistent snapshots "
                "across concurrent transactions.",
                fontsize=12,
            )
    doc.save(pdf)
    doc.close()
    import hashlib

    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    output = run_sdk(pdf)
    assert output["pages"] == 3 and output["raw_pages_equal"]
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == digest
    reopened = pymupdf.open(pdf)
    assert len(reopened[0].get_images()) == 1
    reopened.close()

    def walk(nodes):
        for n in nodes:
            yield n
            yield from walk(n.get("nodes", []))

    assert "Transaction layout" in {n["title"] for n in walk(output["after"])}


@pytest.mark.parametrize(
    "fake",
    [
        "Bullet item:",
        "1. Numbered item",
        "- Unordered item",
        "An inline sentence.",
        "lowercase fragment",
    ],
)
def test_bold_inline_and_list_items_are_not_promoted(fake):
    source, pages = example()
    pages[4]["lines"][3]["text"] = fake
    pages[4]["lines"][3]["bold"] = True
    repaired = repair_structure(source, pages)
    assert fake not in [n["title"] for n in repaired["structure"][0].get("nodes", [])]


def test_existing_nested_children_and_metadata_survive():
    source, pages = example()
    source["structure"][0]["nodes"] = [
        dict(
            title="Existing details",
            node_id="known-child",
            start_index=5,
            end_index=6,
            summary="untouched",
        )
    ]
    pages[4]["lines"].insert(2, line("Existing details", 558, 545.25, 13))
    repaired = repair_structure(source, pages)
    assert [n["title"] for n in repaired["structure"][0]["nodes"]] == [
        "Existing details",
        "Data structures",
        "Scaling Scion",
    ]
    assert repaired["structure"][0]["nodes"][0]["summary"] == "untouched"


def test_malformed_pdf_is_bounded_and_restores_seam(tmp_path):
    pdf = tmp_path / "malformed.pdf"
    pdf.write_bytes(b"%PDF-1.7\ninvalid truncated body")
    assert run_sdk(pdf, "--malformed") == {"rejected": True, "restored": True}


def test_noise_filter_does_not_require_long_body_text():
    source = dict(
        structure=[dict(title="Join the mailing list", node_id="cta", start_index=1, end_index=1)],
        page_texts=["raw footer"],
    )
    pages = [dict(number=1, height=800, lines=[line("Join the mailing list", 60, 45, 15)])]
    assert repair_structure(source, pages)["structure"] == []


def test_repair_is_idempotent():
    source, pages = example()
    repaired = repair_structure(source, pages)
    assert repair_structure(repaired, pages) == repaired
