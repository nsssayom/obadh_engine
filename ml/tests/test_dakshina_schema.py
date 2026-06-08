from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.dakshina import read_lexicon_tsv
from obadh_ml.data.audit import AuditConfig, PairRecord, audit_pair, audit_records
from obadh_ml.schema import FEATURE_SCHEMA_VERSION, feature_keys, require_feature_document


class DakshinaReaderTests(unittest.TestCase):
    def test_reads_native_latin_weight_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bn.translit.sampled.train.tsv"
            path.write_text("কেমন\tkmn\t2\nঅ্যাপ\tapp\t1\n", encoding="utf-8")

            rows = list(read_lexicon_tsv(path))

        self.assertEqual(rows[0].latin, "kmn")
        self.assertEqual(rows[0].target, "কেমন")
        self.assertEqual(rows[0].weight, 2)
        self.assertEqual(rows[1].latin, "app")


class FeatureSchemaTests(unittest.TestCase):
    def test_validates_and_extracts_feature_keys(self) -> None:
        document = {
            "schema": FEATURE_SCHEMA_VERSION,
            "accepted": True,
            "tokens": [
                {
                    "token_type": "word",
                    "slots": [
                        {"feature_key": "before|consonant:k"},
                        {"feature_key": "main|consonant:k"},
                    ],
                }
            ],
        }

        require_feature_document(document)
        self.assertEqual(
            feature_keys(document),
            ["before|consonant:k", "main|consonant:k"],
        )

    def test_rejects_unknown_schema(self) -> None:
        with self.assertRaises(ValueError):
            require_feature_document({"schema": "other", "accepted": True, "tokens": []})


class PairAuditTests(unittest.TestCase):
    def test_accepts_clean_word_pair(self) -> None:
        result = audit_pair(
            PairRecord(source_id="fixture", row_id="1", latin="bhalo", target="ভালো"),
            AuditConfig(mode="word"),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_latin, "bhalo")

    def test_rejects_sentence_rows_for_word_model(self) -> None:
        result = audit_pair(
            PairRecord(
                source_id="fixture",
                row_id="2",
                latin="ami valo achi",
                target="আমি ভালো আছি",
            ),
            AuditConfig(mode="word"),
        )

        self.assertFalse(result.accepted)
        self.assertIn("not_word_pair", {issue.code for issue in result.issues})

    def test_rejects_mixed_target_ascii_for_word_model(self) -> None:
        result = audit_pair(
            PairRecord(
                source_id="fixture",
                row_id="3",
                latin="indicator.apk",
                target="indicator.apk",
            ),
            AuditConfig(mode="word"),
        )

        self.assertFalse(result.accepted)
        self.assertIn("target_contains_ascii_letters", {issue.code for issue in result.issues})

    def test_reports_duplicate_and_conflicting_labels(self) -> None:
        _, summary = audit_records(
            [
                PairRecord(source_id="fixture", row_id="1", latin="bhalo", target="ভালো"),
                PairRecord(source_id="fixture", row_id="2", latin="bhalo", target="ভালো"),
                PairRecord(source_id="fixture", row_id="3", latin="bhalo", target="ভাল"),
            ],
            AuditConfig(mode="word"),
        )

        self.assertEqual(summary.duplicate_latin_targets, 1)
        self.assertEqual(summary.conflicting_latin_labels, 1)

    def test_audit_cli_writes_report_and_accepted_rows(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "pairs.tsv"
            report_path = tmp_path / "report.json"
            accepted_path = tmp_path / "accepted.jsonl"
            input_path.write_text(
                "roman\tbangla\nbhalo\tভালো\nami bhalo\tআমি ভালো\n",
                encoding="utf-8",
            )

            completed = run(
                [
                    sys.executable,
                    str(repo_root / "ml/scripts/audit_pairs.py"),
                    "--input",
                    str(input_path),
                    "--format",
                    "tsv",
                    "--latin-column",
                    "roman",
                    "--target-column",
                    "bangla",
                    "--source-id",
                    "fixture",
                    "--mode",
                    "word",
                    "--report",
                    str(report_path),
                    "--accepted-output",
                    str(accepted_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"accepted_rows":1', completed.stdout)
            self.assertTrue(report_path.exists())
            self.assertEqual(len(accepted_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
