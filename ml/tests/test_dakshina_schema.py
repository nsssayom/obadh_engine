from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.dakshina import read_lexicon_tsv
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


if __name__ == "__main__":
    unittest.main()
