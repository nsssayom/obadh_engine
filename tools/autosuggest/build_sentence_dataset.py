#!/usr/bin/env python3
"""Build sentence-order Bangla corpora for Obadh autosuggestion.

The output is a compact, stream-friendly dataset for next-word modeling:

    source<TAB>document_id<TAB>sentence_id<TAB>token_count<TAB>tokens

`tokens` is a single space-separated Bangla token sequence. Raw EPUB, Kaggle
JSON, and local cache files stay outside the repository; the generated corpus
under data/autosuggest can be audited and rebuilt deterministically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import posixpath
import re
import shutil
import sqlite3
import sys
import unicodedata
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from xml.etree import ElementTree


DEFAULT_WIKI_DIR = Path(
    "~/.cache/kagglehub/datasets/hurutta/bangla-wikipedia-dataset/versions/1/wiki_bn_articles"
).expanduser()
DEFAULT_NEWS_JSON = Path(
    "~/.cache/kagglehub/datasets/furcifer/bangla-newspaper-dataset/versions/2/data_v2/data_v2.json"
).expanduser()
DEFAULT_OUTPUT = Path("data/autosuggest/corpus")

SENTENCE_BOUNDARY_RE = re.compile(r"[।!?]+|(?<!\d)[.](?!\d)|[\r\n]+")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
HASANT = "\u09cd"
KHANDA_TA = "\u09ce"
TA = "\u09a4"


@dataclass
class SourceStats:
    source: str
    documents: int = 0
    source_bytes: int = 0
    emitted_sentences: int = 0
    emitted_tokens: int = 0
    skipped_short_sentences: int = 0
    skipped_long_sentence_chunks: int = 0
    duplicate_sentences: int = 0


@dataclass
class DocumentStats:
    source: str
    document_id: str
    title: str
    source_ref: str
    source_bytes: int
    emitted_sentences: int
    emitted_tokens: int


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_bangla_candidate(word: str) -> str:
    word = normalize(word)

    # Bengali text often uses a terminal joiner after a visible final hasant
    # to prevent accidental shaping. It is orthographic control, not a token
    # identity signal for the autosuggest corpus.
    while len(word) >= 2 and is_joiner(word[-1]) and word[-2] == HASANT:
        word = word[:-1]

    # Normalize the legacy spelling convention ত্ to the dedicated khanda-ta
    # form used by modern Bangla text: বিদ্যুত্ -> বিদ্যুৎ, হঠাত্ -> হঠাৎ.
    if word.endswith(TA + HASANT):
        word = word[: -len(TA + HASANT)] + KHANDA_TA

    return word


def is_joiner(ch: str) -> bool:
    return ch in ("\u200c", "\u200d")


def is_bangla_base_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0985 <= code <= 0x098C
        or 0x098F <= code <= 0x0990
        or 0x0993 <= code <= 0x09A8
        or 0x09AA <= code <= 0x09B0
        or code == 0x09B2
        or 0x09B6 <= code <= 0x09B9
        or code == 0x09CE
        or 0x09DC <= code <= 0x09DD
        or 0x09DF <= code <= 0x09E1
    )


def is_bangla_word_char(ch: str) -> bool:
    code = ord(ch)
    return (
        is_bangla_base_char(ch)
        or 0x0981 <= code <= 0x0983
        or code == 0x09BC
        or 0x09BE <= code <= 0x09C4
        or 0x09C7 <= code <= 0x09C8
        or 0x09CB <= code <= 0x09CD
        or code == 0x09D7
        or 0x09E2 <= code <= 0x09E3
        or code == 0x09FE
    )


def is_assamese_only_letter(ch: str) -> bool:
    return ch in ("\u09f0", "\u09f1")


def is_bangla_token_char(ch: str) -> bool:
    return is_bangla_word_char(ch) or is_joiner(ch) or is_assamese_only_letter(ch)


def is_bangla_word(word: str) -> bool:
    word = normalize_bangla_candidate(word)
    if not word:
        return False

    has_base = False
    base_count = 0
    previous_joiner = False
    previous_hasant = False

    for index, ch in enumerate(word):
        if is_assamese_only_letter(ch):
            return False
        if is_joiner(ch):
            if index == 0 or previous_joiner:
                return False
            previous_joiner = True
            continue
        if not is_bangla_word_char(ch):
            return False
        if not has_base and not is_bangla_base_char(ch):
            return False
        if previous_hasant and not is_bangla_base_char(ch):
            return False
        if previous_hasant and ch == "\u09cd":
            return False

        previous_joiner = False
        previous_hasant = ch == HASANT
        if is_bangla_base_char(ch):
            base_count += 1
            has_base = True

    return has_base and not previous_joiner and (not previous_hasant or base_count >= 2)


def bangla_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []

    for ch in normalize(text):
        if is_bangla_token_char(ch):
            current.append(ch)
            continue
        if current:
            word = normalize_bangla_candidate("".join(current))
            if is_bangla_word(word):
                tokens.append(word)
            current.clear()

    if current:
        word = normalize_bangla_candidate("".join(current))
        if is_bangla_word(word):
            tokens.append(word)

    return tokens


def strip_markup(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def sentence_spans(text: str) -> Iterator[str]:
    for part in SENTENCE_BOUNDARY_RE.split(normalize(text)):
        part = SPACE_RE.sub(" ", part).strip()
        if part:
            yield part


def stable_id(source: str, ref: str) -> str:
    digest = hashlib.blake2b(ref.encode("utf-8"), digest_size=8).hexdigest()
    return f"{source}:{digest}"


def sentence_hash(tokens: list[str]) -> bytes:
    joined = "\u241f".join(tokens)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).digest()


class DuplicateIndex:
    def mark_seen(self, key: bytes) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MemoryDuplicateIndex(DuplicateIndex):
    def __init__(self) -> None:
        self.seen: set[bytes] = set()

    def mark_seen(self, key: bytes) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


class SqliteDuplicateIndex(DuplicateIndex):
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode = OFF")
        self.connection.execute("PRAGMA synchronous = OFF")
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self.connection.execute("CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID")

    def mark_seen(self, key: bytes) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (key,))
        return cursor.rowcount == 1

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class DatasetWriter:
    def __init__(self, output: Path, dedupe_backend: str) -> None:
        self.output = output
        self.sentences_dir = output / "sentences"
        self.build_dir = output / "_build"
        self.output.mkdir(parents=True, exist_ok=True)
        self.sentences_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = output / "documents.tsv.gz"
        self.documents_handle = gzip.open(self.documents_path, "wt", encoding="utf-8", newline="")
        self.documents = csv.writer(self.documents_handle, delimiter="\t", lineterminator="\n")
        self.documents.writerow(
            [
                "source",
                "document_id",
                "title",
                "source_ref",
                "source_bytes",
                "emitted_sentences",
                "emitted_tokens",
            ]
        )
        self.sentence_handles: dict[str, Any] = {}
        self.sentence_writers: dict[str, csv.writer] = {}
        self.duplicate_index: DuplicateIndex | None = self.make_duplicate_index(dedupe_backend)

    def make_duplicate_index(self, dedupe_backend: str) -> DuplicateIndex | None:
        if dedupe_backend == "none":
            return None
        if dedupe_backend == "memory":
            return MemoryDuplicateIndex()
        if dedupe_backend == "sqlite":
            return SqliteDuplicateIndex(self.build_dir / "seen.sqlite")
        raise ValueError(f"unknown dedupe backend: {dedupe_backend}")

    def sentence_writer(self, source: str) -> csv.writer:
        writer = self.sentence_writers.get(source)
        if writer is not None:
            return writer

        path = self.sentences_dir / f"{source}.tsv.gz"
        handle = gzip.open(path, "wt", encoding="utf-8", newline="")
        self.sentence_handles[source] = handle
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source", "document_id", "sentence_id", "token_count", "tokens"])
        self.sentence_writers[source] = writer
        return writer

    def emit_sentence(
        self,
        source: str,
        document_id: str,
        sentence_id: int,
        tokens: list[str],
    ) -> bool:
        if self.duplicate_index is not None:
            key = sentence_hash(tokens)
            if not self.duplicate_index.mark_seen(key):
                return False

        self.sentence_writer(source).writerow(
            [source, document_id, sentence_id, len(tokens), " ".join(tokens)]
        )
        return True

    def emit_document(self, stats: DocumentStats) -> None:
        self.documents.writerow(
            [
                stats.source,
                stats.document_id,
                stats.title,
                stats.source_ref,
                stats.source_bytes,
                stats.emitted_sentences,
                stats.emitted_tokens,
            ]
        )

    def close(self) -> None:
        for handle in self.sentence_handles.values():
            handle.close()
        self.documents_handle.close()
        if self.duplicate_index is not None:
            self.duplicate_index.close()
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)


def emit_document_sentences(
    writer: DatasetWriter,
    source_stats: SourceStats,
    source: str,
    document_id: str,
    source_ref: str,
    title: str,
    text_blocks: Iterable[str],
    source_bytes: int,
    min_tokens: int,
    max_tokens: int,
) -> None:
    source_stats.documents += 1
    source_stats.source_bytes += source_bytes
    sentence_id = 0
    emitted_sentences = 0
    emitted_tokens = 0

    for block in text_blocks:
        for sentence in sentence_spans(block):
            tokens = bangla_tokens(sentence)
            if len(tokens) < min_tokens:
                source_stats.skipped_short_sentences += 1
                continue
            for chunk_start in range(0, len(tokens), max_tokens):
                chunk = tokens[chunk_start : chunk_start + max_tokens]
                if len(chunk) < min_tokens:
                    source_stats.skipped_short_sentences += 1
                    continue
                if chunk_start > 0:
                    source_stats.skipped_long_sentence_chunks += 1
                sentence_id += 1
                if writer.emit_sentence(source, document_id, sentence_id, chunk):
                    emitted_sentences += 1
                    emitted_tokens += len(chunk)
                else:
                    source_stats.duplicate_sentences += 1

    source_stats.emitted_sentences += emitted_sentences
    source_stats.emitted_tokens += emitted_tokens
    writer.emit_document(
        DocumentStats(
            source=source,
            document_id=document_id,
            title=title,
            source_ref=source_ref,
            source_bytes=source_bytes,
            emitted_sentences=emitted_sentences,
            emitted_tokens=emitted_tokens,
        )
    )


def epub_text_blocks(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        member_names = epub_spine_member_names(archive)
        if not member_names:
            member_names = sorted(
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and Path(name.lower()).suffix in (".xhtml", ".html", ".htm", ".txt")
            )

        blocks: list[str] = []
        for name in member_names:
            try:
                raw = archive.read(name).decode("utf-8", errors="replace")
            except KeyError:
                continue
            if Path(name.lower()).suffix == ".txt":
                blocks.append(raw)
            else:
                blocks.append(strip_markup(raw))
        return blocks


def epub_spine_member_names(archive: zipfile.ZipFile) -> list[str]:
    try:
        container = archive.read("META-INF/container.xml")
    except KeyError:
        return []

    try:
        root = ElementTree.fromstring(container)
    except ElementTree.ParseError:
        return []

    opf_path = ""
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "rootfile":
            opf_path = element.attrib.get("full-path", "")
            break
    if not opf_path:
        return []

    try:
        opf = archive.read(opf_path)
        opf_root = ElementTree.fromstring(opf)
    except (KeyError, ElementTree.ParseError):
        return []

    manifest: dict[str, tuple[str, str, str]] = {}
    for element in opf_root.iter():
        if element.tag.rsplit("}", 1)[-1] != "item":
            continue
        item_id = element.attrib.get("id")
        href = element.attrib.get("href")
        if not item_id or not href:
            continue
        media_type = element.attrib.get("media-type", "")
        suffix = Path(href.lower()).suffix
        if media_type not in ("application/xhtml+xml", "text/html", "text/plain") and suffix not in (
            ".xhtml",
            ".html",
            ".htm",
            ".txt",
        ):
            continue
        href = unquote(href.split("#", 1)[0])
        full_path = posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), href))
        manifest[item_id] = (
            full_path,
            media_type,
            element.attrib.get("properties", ""),
        )

    names: list[str] = []
    for element in opf_root.iter():
        if element.tag.rsplit("}", 1)[-1] != "itemref":
            continue
        if element.attrib.get("linear", "").lower() == "no":
            continue
        idref = element.attrib.get("idref", "")
        item = manifest.get(idref)
        if item is None:
            continue
        if "nav" in item[2].split():
            continue
        names.append(item[0])

    return list(dict.fromkeys(names))


def iter_epub_documents(epub_dir: Path) -> Iterator[tuple[str, str, str, list[str], int]]:
    for path in sorted(epub_dir.glob("*.epub")):
        source_ref = str(path)
        title = path.stem
        yield (
            stable_id("epub", source_ref),
            source_ref,
            title,
            epub_text_blocks(path),
            path.stat().st_size,
        )


def iter_wiki_documents(wiki_dir: Path) -> Iterator[tuple[str, str, str, list[str], int]]:
    for path in sorted(wiki_dir.glob("*.json")):
        source_bytes = path.stat().st_size
        try:
            article = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "")
        content = str(article.get("content") or "")
        pageid = article.get("pageid", path.stem)
        source_ref = str(article.get("url") or path)
        yield (
            stable_id("wiki", str(pageid)),
            source_ref,
            title,
            [title, content],
            source_bytes,
        )


def require_ijson():
    try:
        import ijson  # type: ignore
    except ImportError as error:
        raise SystemExit("ijson is required for streaming the newspaper JSON dataset") from error
    return ijson


def iter_news_documents(news_json: Path, max_documents: int | None) -> Iterator[tuple[str, str, str, list[str], int]]:
    ijson = require_ijson()
    source_bytes = news_json.stat().st_size
    with news_json.open("rb") as handle:
        for index, item in enumerate(ijson.items(handle, "item"), start=1):
            if max_documents is not None and index > max_documents:
                break
            if not isinstance(item, dict):
                continue
            title = text_value(item.get("title"))
            content = text_value(item.get("content"))
            if not title and not content:
                continue
            source_ref = text_value(item.get("url")) or f"{news_json}#{index}"
            document_key = text_value(item.get("id")) or source_ref
            yield (
                stable_id("news", document_key),
                source_ref,
                title,
                [title, content],
                source_bytes if index == 1 else 0,
            )


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(item for item in value if isinstance(item, str))
    return ""


def write_manifest(output: Path, args: argparse.Namespace, stats: list[SourceStats]) -> None:
    manifest = {
        "version": 1,
        "format": "source<TAB>document_id<TAB>sentence_id<TAB>token_count<TAB>tokens",
        "normalization": "Unicode NFC",
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "dedupe_backend": args.dedupe_backend,
        "sources": [asdict(stat) for stat in stats],
        "paths": {
            "sentences": "sentences/{source}.tsv.gz",
            "documents": "documents.tsv.gz",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_source(
    writer: DatasetWriter,
    source_stats: SourceStats,
    documents: Iterable[tuple[str, str, str, list[str], int]],
    min_tokens: int,
    max_tokens: int,
    progress_every: int,
) -> None:
    for index, (document_id, source_ref, title, blocks, source_bytes) in enumerate(documents, start=1):
        emit_document_sentences(
            writer,
            source_stats,
            source_stats.source,
            document_id,
            source_ref,
            title,
            blocks,
            source_bytes,
            min_tokens,
            max_tokens,
        )
        if progress_every > 0 and index % progress_every == 0:
            print(asdict(source_stats), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epub-dir", type=Path, default=Path("epubs"))
    parser.add_argument("--wiki-dir", type=Path, default=DEFAULT_WIKI_DIR)
    parser.add_argument("--news-json", type=Path, default=DEFAULT_NEWS_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", action="append", choices=("epub", "wiki", "news"))
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--news-max-documents", type=int)
    parser.add_argument("--dedupe-backend", choices=("sqlite", "memory", "none"), default="sqlite")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    if not args.dedupe:
        args.dedupe_backend = "none"

    sources = set(args.source or ("epub", "wiki", "news"))
    writer = DatasetWriter(args.output, dedupe_backend=args.dedupe_backend)
    stats: list[SourceStats] = []

    try:
        if "epub" in sources:
            source_stats = SourceStats("epub")
            process_source(
                writer,
                source_stats,
                iter_epub_documents(args.epub_dir),
                args.min_tokens,
                args.max_tokens,
                args.progress_every,
            )
            stats.append(source_stats)

        if "wiki" in sources:
            source_stats = SourceStats("wiki")
            process_source(
                writer,
                source_stats,
                iter_wiki_documents(args.wiki_dir),
                args.min_tokens,
                args.max_tokens,
                args.progress_every,
            )
            stats.append(source_stats)

        if "news" in sources:
            source_stats = SourceStats("news")
            process_source(
                writer,
                source_stats,
                iter_news_documents(args.news_json, args.news_max_documents),
                args.min_tokens,
                args.max_tokens,
                args.progress_every,
            )
            stats.append(source_stats)
    finally:
        writer.close()

    write_manifest(args.output, args, stats)
    print(json.dumps({"output": str(args.output), "sources": [asdict(s) for s in stats]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
