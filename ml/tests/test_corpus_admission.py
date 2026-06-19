from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CalledProcessError, run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.corpus import ADMITTED_PAIR_SCHEMA, CORPUS_SUMMARY_SCHEMA
from obadh_ml.data.sources import source_spec_from_json


class CorpusAdmissionTests(unittest.TestCase):
    def test_source_spec_rejects_excluded_sources(self) -> None:
        with self.assertRaises(ValueError):
            source_spec_from_json(
                {
                    "id": "aksharantar",
                    "kind": "generic_pairs",
                    "path": "ignored.csv",
                }
            )

    def test_prepare_corpus_writes_reports_and_admitted_rows(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dakshina_root = tmp_path / "dakshina"
            lexicon_dir = dakshina_root / "bn" / "lexicons"
            lexicon_dir.mkdir(parents=True)
            (lexicon_dir / "bn.translit.sampled.train.tsv").write_text(
                "ভালো\tbhalo\t2\nআমি ami\tami ami\t1\n",
                encoding="utf-8",
            )

            banglatlit_path = tmp_path / "banglatlit.csv"
            banglatlit_path.write_text(
                "id,text_transliterated,text_bengali\n"
                "row-1,ami bhalo,আমি ভালো\n"
                "row-2,indicator.apk,indicator.apk\n",
                encoding="utf-8",
            )

            manifest_path = tmp_path / "manifest.json"
            output_dir = tmp_path / "out"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "obadh.ml.corpus_manifest.v0",
                        "corpus_id": "fixture_corpus",
                        "sources": [
                            {
                                "id": "dakshina_fixture_train",
                                "kind": "dakshina_bn",
                                "path": str(dakshina_root),
                                "split": "train",
                                "mode": "word",
                                "admit": True,
                            },
                            {
                                "id": "banglatlit_fixture",
                                "kind": "banglatlit",
                                "path": str(banglatlit_path),
                                "split": "train",
                                "mode": "sentence",
                                "admit": False,
                                "audit": {
                                    "max_latin_chars": 200,
                                    "max_target_chars": 200,
                                    "allow_sentence_punctuation": True,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            completed = run(
                [
                    sys.executable,
                    str(repo_root / "ml/scripts/prepare_corpus.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary_path = Path(completed.stdout.strip())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], CORPUS_SUMMARY_SCHEMA)
            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["admitted_sources"], 1)
            self.assertEqual(summary["accepted_rows"], 1)

            accepted_files = list((output_dir / "accepted").glob("*.jsonl"))
            self.assertEqual(len(accepted_files), 1)
            admitted = json.loads(accepted_files[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(admitted["schema"], ADMITTED_PAIR_SCHEMA)
            self.assertEqual(admitted["latin"], "bhalo")
            self.assertEqual(admitted["target"], "ভালো")

            report_files = list((output_dir / "reports").glob("*.audit.json"))
            self.assertEqual(len(report_files), 2)

    def test_prepare_corpus_blocks_dakshina_benchmark_admission(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dakshina_root = tmp_path / "dakshina"
            lexicon_dir = dakshina_root / "bn" / "lexicons"
            lexicon_dir.mkdir(parents=True)
            (lexicon_dir / "bn.translit.sampled.dev.tsv").write_text(
                "ভালো\tbhalo\t1\n",
                encoding="utf-8",
            )
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "obadh.ml.corpus_manifest.v0",
                        "corpus_id": "fixture_corpus",
                        "sources": [
                            {
                                "id": "dakshina_fixture_dev",
                                "kind": "dakshina_bn",
                                "path": str(dakshina_root),
                                "split": "dev",
                                "mode": "word",
                                "admit": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(CalledProcessError):
                run(
                    [
                        sys.executable,
                        str(repo_root / "ml/scripts/prepare_corpus.py"),
                        "--manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(tmp_path / "out"),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
