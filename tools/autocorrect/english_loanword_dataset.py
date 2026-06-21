#!/usr/bin/env python3
"""Build and audit English-spelling to Bengali-loanword seed candidates.

This is an offline data investigation tool. It does not participate in runtime
autocorrection and should not be used as a word-specific correction table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


KAIKKI_BENGALI_JSONL_URL = (
    "https://kaikki.org/dictionary/Bengali/kaikki.org-dictionary-Bengali.jsonl"
)
LOANWORDBANK_BENGALI_CSV_URL = (
    "https://raw.githubusercontent.com/loanwordbank/wiktionary_cldf/master/raw2/"
    "bengali.csv"
)
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"

BANGLA_RE = re.compile(r"[\u0980-\u09ff]")
BANGLA_LETTER_RE = re.compile(r"[\u0985-\u09b9\u09dc-\u09df]")
ENGLISH_KEY_RE = re.compile(r"[A-Za-z][A-Za-z .'\-]{0,80}")
SINGLE_TOKEN_KEY_RE = re.compile(r"[a-z][a-z'\-]{1,40}")
BORROWED_TEXT_RE = re.compile(
    r"(?:Borrowed from|borrowed from) English ([A-Za-z][A-Za-z .'\-’]{0,80})"
)
TRANSLIT_TEXT_RE = re.compile(
    r"(?:Transliteration of|transliteration of) English "
    r"([A-Za-z][A-Za-z .'\-’]{0,80})"
)

DIRECT_ENGLISH_TEMPLATE_NAMES = {
    "bor",
    "bor+",
    "borrowing",
    "der",
    "der+",
    "lbor",
    "translit",
    "uder",
}


@dataclass
class Candidate:
    key: str
    word: str
    frequency: int
    evidence: set[str] = field(default_factory=set)

    @property
    def source_families(self) -> set[str]:
        return {item.split(":", 1)[0] for item in self.evidence}

    @property
    def has_kaikki_template(self) -> bool:
        return any(item.startswith("kaikki_template:") for item in self.evidence)

    @property
    def has_kaikki_text(self) -> bool:
        return any(item.startswith("kaikki_text:") for item in self.evidence)

    @property
    def has_loanwordbank(self) -> bool:
        return "loanwordbank:etym" in self.evidence

    @property
    def confidence(self) -> int:
        score = 0
        if self.has_kaikki_template:
            score += 75
        if self.has_kaikki_text:
            score += 12
        if self.has_loanwordbank:
            score += 10
        if len(self.source_families) >= 2:
            score += 8
        if self.frequency >= 1_000:
            score += 3
        return min(score, 100)

    @property
    def tier(self) -> str:
        if self.confidence >= 90:
            return "accept"
        if self.confidence >= 75:
            return "review"
        return "weak"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Investigate English loanword dataset sources for Obadh."
    )
    parser.add_argument(
        "--bangla-lexicon",
        default="data/autocorrect/lexicons/merged/bn.tsv",
        type=Path,
        help="Current Bengali word-frequency TSV used as the valid-word gate.",
    )
    parser.add_argument(
        "--workdir",
        default="data/autocorrect/tmp/english_loanwords",
        type=Path,
        help="Ignored directory for downloaded sources and generated audits.",
    )
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        metavar="english=বাংলা",
        help="Expected pair to measure source coverage. Can be passed repeatedly.",
    )
    parser.add_argument(
        "--skeleton-probes",
        action="store_true",
        help="Run experimental offline spelling-skeleton retrieval for probes.",
    )
    parser.add_argument(
        "--skeleton-min-frequency",
        default=2,
        type=int,
        help="Minimum Bengali lexicon frequency admitted into the skeleton index.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use only already-downloaded source files.",
    )
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    source_paths = source_files(args.workdir, download=not args.no_download)
    lexicon = read_bangla_lexicon(args.bangla_lexicon)
    probes = parse_probe_args(args.probe)

    direct_pairs, raw_stats = collect_direct_candidates(
        source_paths["kaikki"], source_paths["loanwordbank"], lexicon
    )
    cmudict_entries = read_cmudict(source_paths["cmudict"])
    skeleton_index = (
        build_skeleton_index(lexicon, args.skeleton_min_frequency)
        if probes and args.skeleton_probes
        else {}
    )

    candidates = sorted(
        direct_pairs.values(),
        key=lambda item: (-item.confidence, -item.frequency, item.key, item.word),
    )
    seed_path = args.workdir / "loanword_seed_candidates.tsv"
    write_seed_tsv(seed_path, candidates)
    accepted_path = args.workdir / "loanword_accept_candidates.tsv"
    write_seed_tsv(
        accepted_path,
        [candidate for candidate in candidates if candidate.tier == "accept"],
    )

    probe_rows = probe_coverage(
        probes, direct_pairs, lexicon, cmudict_entries, skeleton_index
    )
    gaps_path = args.workdir / "loanword_probe_gaps.tsv"
    write_probe_gaps(gaps_path, probe_rows)

    report = build_report(
        lexicon_size=len(lexicon),
        raw_stats=raw_stats,
        candidates=candidates,
        cmudict_entries=cmudict_entries,
        probes=probe_rows,
        skeleton_probes=args.skeleton_probes,
        skeleton_min_frequency=args.skeleton_min_frequency,
        seed_path=seed_path,
        accepted_path=accepted_path,
        gaps_path=gaps_path,
        source_paths=source_paths,
    )
    report_path = args.workdir / "loanword_investigation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def source_files(workdir: Path, download: bool) -> dict[str, Path]:
    paths = {
        "kaikki": workdir / "kaikki-bn.jsonl",
        "loanwordbank": workdir / "loanwordbank-bengali.csv",
        "cmudict": workdir / "cmudict.dict",
    }
    urls = {
        "kaikki": KAIKKI_BENGALI_JSONL_URL,
        "loanwordbank": LOANWORDBANK_BENGALI_CSV_URL,
        "cmudict": CMUDICT_URL,
    }
    if download:
        for key, path in paths.items():
            if not path.exists():
                download_file(urls[key], path)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing source files: {', '.join(missing)}")
    return paths


def download_file(url: str, output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".download")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(output)


def read_bangla_lexicon(path: Path) -> dict[str, int]:
    lexicon: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        word = columns[0].strip()
        if not is_bangla_word(word):
            continue
        frequency = 1
        if len(columns) > 1 and columns[1].strip().isdigit():
            frequency = int(columns[1].strip())
        lexicon[word] = lexicon.get(word, 0) + frequency
    return lexicon


def collect_direct_candidates(
    kaikki_path: Path,
    loanwordbank_path: Path,
    lexicon: dict[str, int],
) -> tuple[dict[tuple[str, str], Candidate], dict[str, int]]:
    candidates: dict[tuple[str, str], Candidate] = {}
    stats: dict[str, int] = defaultdict(int)

    for item in read_kaikki_jsonl(kaikki_path):
        stats["kaikki_rows"] += 1
        word = str(item.get("word", "")).strip()
        if not is_bangla_word(word):
            stats["kaikki_non_bangla_words"] += 1
            continue
        if word not in lexicon:
            stats["kaikki_target_not_in_lexicon"] += 1

        for key, evidence in english_etymons_from_kaikki(item):
            add_candidate(candidates, key, word, lexicon, evidence, stats)

    with loanwordbank_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["loanwordbank_rows"] += 1
            word = row.get("L2_orth", "").strip()
            if not is_bangla_word(word):
                stats["loanwordbank_non_bangla_words"] += 1
                continue
            key = clean_english_key(row.get("L2_etym", ""))
            if key is None:
                stats["loanwordbank_missing_or_invalid_etymon"] += 1
                continue
            add_candidate(candidates, key, word, lexicon, "loanwordbank:etym", stats)

    return candidates, dict(sorted(stats.items()))


def read_kaikki_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def english_etymons_from_kaikki(item: dict) -> Iterable[tuple[str, str]]:
    for template in item.get("etymology_templates", []) or []:
        name = str(template.get("name", ""))
        args = template.get("args") or {}
        if name not in DIRECT_ENGLISH_TEMPLATE_NAMES:
            continue
        if args.get("2") != "en":
            continue
        key = clean_english_key(str(args.get("3", "")))
        if key is not None:
            yield key, f"kaikki_template:{name}"

    etymology_text = str(item.get("etymology_text", ""))
    for match in BORROWED_TEXT_RE.finditer(etymology_text):
        key = clean_english_key(match.group(1).replace("’", "'"))
        if key is not None:
            yield key, "kaikki_text:borrowed"
    for match in TRANSLIT_TEXT_RE.finditer(etymology_text):
        key = clean_english_key(match.group(1).replace("’", "'"))
        if key is not None:
            yield key, "kaikki_text:translit"


def add_candidate(
    candidates: dict[tuple[str, str], Candidate],
    key: str,
    word: str,
    lexicon: dict[str, int],
    evidence: str,
    stats: dict[str, int],
) -> None:
    stats["raw_pairs"] += 1
    if not SINGLE_TOKEN_KEY_RE.fullmatch(key):
        stats["non_single_token_keys"] += 1
        return
    if word not in lexicon:
        stats["target_not_in_lexicon_pairs"] += 1
        return
    pair_key = (key, word)
    candidate = candidates.get(pair_key)
    if candidate is None:
        candidate = Candidate(key=key, word=word, frequency=lexicon[word])
        candidates[pair_key] = candidate
    candidate.evidence.add(evidence)


def clean_english_key(value: str) -> str | None:
    key = value.strip().strip(".,;:()[]{}\"“”‘’")
    key = re.sub(r"\s+", " ", key)
    if not ENGLISH_KEY_RE.fullmatch(key):
        return None
    key = key.lower()
    if key in {"english", "middle english", "old english"}:
        return None
    return key


def is_bangla_word(word: str) -> bool:
    if not BANGLA_RE.search(word) or not BANGLA_LETTER_RE.search(word):
        return False
    for char in word:
        if "\u0980" <= char <= "\u09ff" or char in "\u200c\u200d":
            continue
        return False
    return True


def read_cmudict(path: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(";;;"):
            continue
        columns = line.split()
        if len(columns) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", columns[0]).lower()
        if not SINGLE_TOKEN_KEY_RE.fullmatch(word):
            continue
        entries.setdefault(word, []).append(" ".join(columns[1:]))
    return entries


def parse_probe_args(values: list[str]) -> dict[str, str]:
    probes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --probe {value!r}; expected english=বাংলা")
        key, expected = value.split("=", 1)
        key = clean_english_key(key)
        expected = expected.strip()
        if key is None or not SINGLE_TOKEN_KEY_RE.fullmatch(key):
            raise SystemExit(f"invalid probe English key: {value!r}")
        if not is_bangla_word(expected):
            raise SystemExit(f"invalid probe Bengali word: {value!r}")
        probes[key] = expected
    return probes


def probe_coverage(
    probes: dict[str, str],
    candidates: dict[tuple[str, str], Candidate],
    lexicon: dict[str, int],
    cmudict: dict[str, list[str]],
    skeleton_index: dict[tuple[int, str], list[tuple[str, int, str]]],
) -> list[dict]:
    rows = []
    by_key: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates.values():
        by_key[candidate.key].append(candidate)

    for key, expected in sorted(probes.items()):
        direct_candidates = sorted(
            by_key.get(key, []),
            key=lambda item: (-item.confidence, -item.frequency, item.word),
        )
        skeleton_candidates = skeleton_probe_candidates(key, skeleton_index)
        rows.append(
            {
                "english_key": key,
                "expected_bangla": expected,
                "expected_in_bangla_lexicon": expected in lexicon,
                "direct_source_match": any(
                    candidate.word == expected for candidate in direct_candidates
                ),
                "direct_candidates": [
                    {
                        "word": candidate.word,
                        "frequency": candidate.frequency,
                        "confidence": candidate.confidence,
                        "evidence": sorted(candidate.evidence),
                    }
                    for candidate in direct_candidates[:12]
                ],
                "cmudict_pronunciations": cmudict.get(key, [])[:5],
                "english_skeleton": english_skeleton(key),
                "expected_bangla_skeleton": bangla_skeleton(expected),
                "skeleton_match": any(
                    candidate["word"] == expected for candidate in skeleton_candidates
                ),
                "skeleton_candidates": skeleton_candidates,
            }
        )
    return rows


def build_skeleton_index(
    lexicon: dict[str, int],
    min_frequency: int,
) -> dict[tuple[int, str], list[tuple[str, int, str]]]:
    index: dict[tuple[int, str], list[tuple[str, int, str]]] = defaultdict(list)
    for word, frequency in lexicon.items():
        if frequency < min_frequency:
            continue
        skeleton = bangla_skeleton(word)
        if len(skeleton) < 3:
            continue
        index[(len(skeleton), first_sound_bucket(skeleton))].append((word, frequency, skeleton))
    return dict(index)


BANGLA_SKELETON_MAP = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
    "ং": "ng",
    "ঁ": "n",
    "ঃ": "h",
    "্": "",
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "ch",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "n",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "f",
    "ব": "b",
    "ভ": "v",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "s",
    "ষ": "s",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "rh",
    "য়": "y",
    "ৎ": "t",
}


def bangla_skeleton(word: str) -> str:
    skeleton = "".join(BANGLA_SKELETON_MAP.get(char, "") for char in word)
    for source, target in [
        ("sh", "s"),
        ("kh", "k"),
        ("gh", "g"),
        ("th", "t"),
        ("dh", "d"),
        ("jh", "j"),
    ]:
        skeleton = skeleton.replace(source, target)
    return collapse_repeated(skeleton)


def english_skeleton(word: str) -> str:
    skeleton = re.sub(r"[^a-z]", "", word.lower())
    for source, target in [
        ("tion", "shon"),
        ("sion", "shon"),
        ("ture", "char"),
        ("sure", "shar"),
        ("phone", "fon"),
        ("ware", "war"),
        ("air", "ear"),
        ("eer", "iar"),
        ("ear", "iar"),
        ("ck", "k"),
        ("qu", "k"),
        ("ph", "f"),
        ("sh", "s"),
        ("th", "t"),
        ("gh", "g"),
        ("x", "ks"),
        ("ee", "i"),
        ("ea", "i"),
        ("oo", "u"),
        ("ou", "au"),
        ("ow", "o"),
        ("ai", "e"),
        ("ay", "e"),
        ("ey", "e"),
        ("ie", "i"),
        ("ei", "i"),
        ("oa", "o"),
        ("ue", "u"),
    ]:
        skeleton = skeleton.replace(source, target)
    if skeleton.endswith("e") and len(skeleton) > 3:
        skeleton = skeleton[:-1]

    output = []
    for index, char in enumerate(skeleton):
        next_char = skeleton[index + 1] if index + 1 < len(skeleton) else ""
        if char == "c":
            output.append("s" if next_char in "eiy" else "k")
        elif char == "g":
            output.append("j" if next_char in "eiy" else "g")
        elif char == "y":
            output.append("i")
        elif char == "u" and index == 0:
            output.append("iu")
        else:
            output.append(char)
    return collapse_repeated("".join(output))


def collapse_repeated(value: str) -> str:
    return re.sub(r"(.)\1+", r"\1", value)


def skeleton_probe_candidates(
    key: str,
    skeleton_index: dict[tuple[int, str], list[tuple[str, int, str]]],
    max_distance: int = 3,
    limit: int = 12,
) -> list[dict]:
    if not skeleton_index:
        return []
    source = english_skeleton(key)
    matches = []
    bucket = first_sound_bucket(source)
    for length in range(max(0, len(source) - max_distance), len(source) + max_distance + 1):
        for word, frequency, target in skeleton_index.get((length, bucket), []):
            distance = bounded_levenshtein(source, target, max_distance)
            if distance <= max_distance:
                score = skeleton_candidate_score(distance, frequency)
                matches.append((-score, distance, -frequency, word, frequency, target, score))
    matches.sort()
    return [
        {
            "word": word,
            "frequency": frequency,
            "distance": distance,
            "score": score,
            "skeleton": target,
        }
        for _, distance, _, word, frequency, target, score in matches[:limit]
    ]


def skeleton_candidate_score(distance: int, frequency: int) -> int:
    return round(math.log10(frequency + 1) * 600) - (distance * 1000)


def first_sound_bucket(skeleton: str) -> str:
    if not skeleton:
        return ""
    return "V" if skeleton[0] in "aeiou" else skeleton[0]


def bounded_levenshtein(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index] + [0] * len(right)
        best = current[0]
        for right_index, right_char in enumerate(right, 1):
            cost = 0 if left_char == right_char else 1
            current[right_index] = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            best = min(best, current[right_index])
        if best > limit:
            return limit + 1
        previous = current
    return previous[-1]


def write_seed_tsv(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "english_key",
                "bangla_word",
                "confidence",
                "tier",
                "bangla_frequency",
                "evidence",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.key,
                    candidate.word,
                    candidate.confidence,
                    candidate.tier,
                    candidate.frequency,
                    ",".join(sorted(candidate.evidence)),
                ]
            )


def write_probe_gaps(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "english_key",
                "expected_bangla",
                "expected_in_bangla_lexicon",
                "direct_source_match",
                "cmudict_pronunciations",
                    "direct_candidates",
                    "skeleton_candidates",
                ]
            )
        for row in rows:
            writer.writerow(
                [
                    row["english_key"],
                    row["expected_bangla"],
                    str(row["expected_in_bangla_lexicon"]).lower(),
                    str(row["direct_source_match"]).lower(),
                    " | ".join(row["cmudict_pronunciations"]),
                    json.dumps(row["direct_candidates"], ensure_ascii=False),
                    json.dumps(row["skeleton_candidates"], ensure_ascii=False),
                ]
            )


def build_report(
    *,
    lexicon_size: int,
    raw_stats: dict[str, int],
    candidates: list[Candidate],
    cmudict_entries: dict[str, list[str]],
    probes: list[dict],
    skeleton_probes: bool,
    skeleton_min_frequency: int,
    seed_path: Path,
    accepted_path: Path,
    gaps_path: Path,
    source_paths: dict[str, Path],
) -> dict:
    tier_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    key_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        tier_counts[candidate.tier] += 1
        key_counts[candidate.key] += 1
        for evidence in candidate.evidence:
            evidence_counts[evidence] += 1

    ambiguous_keys = [
        {"english_key": key, "candidate_count": count}
        for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ][:25]
    top_candidates = [
        {
            "english_key": candidate.key,
            "bangla_word": candidate.word,
            "confidence": candidate.confidence,
            "tier": candidate.tier,
            "bangla_frequency": candidate.frequency,
            "evidence": sorted(candidate.evidence),
        }
        for candidate in candidates[:25]
    ]

    return {
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "outputs": {
            "seed_candidates_tsv": str(seed_path),
            "accept_candidates_tsv": str(accepted_path),
            "probe_gaps_tsv": str(gaps_path),
        },
        "bangla_lexicon_words": lexicon_size,
        "cmudict_words": len(cmudict_entries),
        "raw_stats": raw_stats,
        "candidate_count": len(candidates),
        "unique_english_keys": len(key_counts),
        "skeleton_probe_config": {
            "enabled": skeleton_probes,
            "min_bangla_frequency": skeleton_min_frequency if skeleton_probes else None,
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "ambiguous_keys_sample": ambiguous_keys,
        "top_candidates_sample": top_candidates,
        "probe_coverage": probes,
    }


if __name__ == "__main__":
    sys.exit(main())
