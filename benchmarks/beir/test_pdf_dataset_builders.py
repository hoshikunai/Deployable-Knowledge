from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_financebench_dataset import parse_evidence
from build_mmlongbench_dataset import evidence_kinds
from build_ohr_bench_dataset import extract_pdfs, parse_domains
from pdf_dataset_tools import (PdfDocument, PdfQuestion, check_numbering, document_id,
                               evidence_on_page, numbering_match_rates, query_groups, text_layer,
                               write_pdf_dataset)

PAGES = {
    "report": [
        "Annual report cover page with the company name",
        "Revenue grew to 842 million dollars driven by strong loan demand across regions",
        "Operating expenses fell because the branch network was consolidated last year",
    ],
}


def question(identifier: str, evidence: tuple[tuple[str, int], ...]) -> PdfQuestion:
    return PdfQuestion(identifier, "What was revenue?", evidence, "table", "finance", "text", True,
                       {"answer": "842"})


class EvidenceMatchingTests(unittest.TestCase):
    def test_evidence_is_found_on_its_page_only(self) -> None:
        evidence = "revenue grew to 842 million dollars driven by strong loan demand"
        self.assertTrue(evidence_on_page(evidence, PAGES["report"][1]))
        self.assertFalse(evidence_on_page(evidence, PAGES["report"][2]))

    def test_evidence_too_short_for_trigrams_never_matches(self) -> None:
        self.assertFalse(evidence_on_page("842 million", PAGES["report"][1]))

    def test_match_rates_per_numbering_base(self) -> None:
        samples = [("report", 1, "revenue grew to 842 million dollars"),
                   ("report", 2, "operating expenses fell because the branch network")]
        self.assertEqual(numbering_match_rates(samples, PAGES), {0: 1.0, 1: 0.0})

    def test_wrong_published_base_is_refused(self) -> None:
        samples = [("report", 1, "revenue grew to 842 million dollars")]
        self.assertEqual(check_numbering(samples, PAGES, published_base=0), {"base0": 1.0, "base1": 0.0})
        with self.assertRaisesRegex(RuntimeError, "page numbering"):
            check_numbering(samples, PAGES, published_base=1)

    def test_page_numbers_past_the_end_do_not_match(self) -> None:
        samples = [("report", 9, "revenue grew to 842 million dollars")]
        self.assertEqual(numbering_match_rates(samples, PAGES), {0: 0.0, 1: 0.0})


class TextLayerTests(unittest.TestCase):
    def test_blank_evidence_page_is_scanned(self) -> None:
        pages = {"scan": ["Cover", "   "]}
        self.assertEqual(text_layer((("scan", 1),), pages), "text")
        self.assertEqual(text_layer((("scan", 1), ("scan", 2)), pages), "scanned")


class DocumentIdTests(unittest.TestCase):
    def test_unsafe_characters_are_replaced(self) -> None:
        self.assertEqual(document_id("law/Fund POS 8C_EX-99.(H)(3)"), "law-Fund-POS-8C_EX-99.-H-3")

    def test_empty_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            document_id("///")


class QueryGroupsTests(unittest.TestCase):
    def test_table_key_is_left_out_when_the_dataset_does_not_label_tables(self) -> None:
        unlabelled = PdfQuestion("q", "Q?", (("report", 1),), "metrics-generated", "unlabelled",
                                 "text", None, {})
        self.assertNotIn("table", query_groups(unlabelled))
        self.assertEqual(query_groups(question("q", (("report", 1),)))["table"], "yes")


class WriteDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.pdf = self.root / "report.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 report")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_writes_page_and_document_splits(self) -> None:
        output = self.root / "dataset"
        summary = write_pdf_dataset(output, [PdfDocument("report", self.pdf, 3)],
                                    [question("q1", (("report", 2), ("report", 3), ("report", 2)))],
                                    {"source": "test"})
        self.assertEqual(summary["testQrels"], 2)
        self.assertEqual((output / "qrels" / "test.tsv").read_text().splitlines()[1:],
                         ["q1\treport#p2\t1", "q1\treport#p3\t1"])
        self.assertEqual((output / "qrels" / "doc.tsv").read_text().splitlines()[1:], ["q1\treport\t1"])
        corpus = [json.loads(line)["_id"] for line in (output / "corpus.jsonl").read_text().splitlines()]
        self.assertEqual(corpus, ["report", "report#p1", "report#p2", "report#p3"])
        manifest = json.loads((output / "pdf-manifest.json").read_text())
        self.assertEqual(manifest["splitUnits"], {"test": "page", "doc": "document"})
        self.assertEqual(manifest["documents"][0]["file"], "pdfs/report.pdf")
        groups = json.loads((output / "query-groups.json").read_text())
        self.assertEqual(groups["q1"], {"source": "table", "type": "finance", "textLayer": "text",
                                        "table": "yes"})

    def test_page_outside_the_document_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "has 3 pages"):
            write_pdf_dataset(self.root / "dataset", [PdfDocument("report", self.pdf, 3)],
                              [question("q1", (("report", 4),))], {})

    def test_identical_pdfs_are_refused(self) -> None:
        copy = self.root / "copy.pdf"
        copy.write_bytes(self.pdf.read_bytes())
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            write_pdf_dataset(self.root / "dataset",
                              [PdfDocument("report", self.pdf, 3), PdfDocument("copy", copy, 3)],
                              [question("q1", (("report", 1),))], {})


class OhrBenchTests(unittest.TestCase):
    def test_domains(self) -> None:
        self.assertEqual(parse_domains("manual,finance"), ("finance", "manual"))
        self.assertEqual(len(parse_domains("all")), 7)
        with self.assertRaises(Exception):
            parse_domains("finance,cooking")

    def test_extracts_only_chosen_domains_under_document_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "pdfs.zip", "w") as archive:
                archive.writestr("finance/", "")
                archive.writestr("finance/JPM 10Q (a).pdf", b"%PDF-finance")
                archive.writestr("news/paper.pdf", b"%PDF-news")
            extracted = extract_pdfs(root / "pdfs.zip", root, ("finance",))
            self.assertEqual(list(extracted), ["finance/JPM 10Q (a)"])
            self.assertEqual(extracted["finance/JPM 10Q (a)"].name, "finance-JPM-10Q-a.pdf")

    def test_unexpected_archive_layout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "pdfs.zip", "w") as archive:
                archive.writestr("finance/nested/file.pdf", b"%PDF-")
            with self.assertRaisesRegex(RuntimeError, "Unexpected entry"):
                extract_pdfs(root / "pdfs.zip", root, ("finance",))


class FinanceBenchTests(unittest.TestCase):
    def test_evidence_literal_is_parsed(self) -> None:
        record = {"financebench_id": "f1", "evidence": str([{
            "doc_name": "3M_2018_10K", "evidence_page_num": 59, "evidence_text_full_page": "Cash flows"}])}
        self.assertEqual(parse_evidence(record)[0]["evidence_page_num"], 59)

    def test_malformed_evidence_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Malformed evidence"):
            parse_evidence({"financebench_id": "f1", "evidence": "[{'doc_name': '3M_2018_10K'}]"})


class MmLongBenchTests(unittest.TestCase):
    def test_evidence_kinds_are_normalized_and_joined(self) -> None:
        record = {"evidence_sources": "['Table', 'Pure-text (Plain-text)', 'Table']"}
        self.assertEqual(evidence_kinds(record), "table+text")

    def test_missing_evidence_kinds_are_unlabelled(self) -> None:
        self.assertEqual(evidence_kinds({"evidence_sources": "[]"}), "unlabelled")

    def test_unknown_evidence_kind_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unknown evidence sources"):
            evidence_kinds({"evidence_sources": "['Hologram']"})


if __name__ == "__main__":
    unittest.main()
