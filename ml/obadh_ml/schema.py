"""Helpers for the versioned Obadh ML feature document."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

FEATURE_SCHEMA_VERSION = "obadh.ml.features.v0"
FEATURE_SLOTS_PER_UNIT = 3


def require_feature_document(document: Mapping[str, object]) -> None:
    """Validate the parts of the feature document that training code relies on."""

    schema = document.get("schema")
    if schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema: {schema!r}")

    accepted = document.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("feature document must contain boolean 'accepted'")

    tokens = document.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("feature document must contain list 'tokens'")


def iter_word_tokens(document: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    require_feature_document(document)
    for token in document["tokens"]:
        if isinstance(token, Mapping) and token.get("token_type") == "word":
            yield token


def iter_slots(document: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    for token in iter_word_tokens(document):
        slots = token.get("slots", [])
        if isinstance(slots, Iterable):
            for slot in slots:
                if isinstance(slot, Mapping):
                    yield slot


def feature_keys(document: Mapping[str, object]) -> list[str]:
    return [str(slot["feature_key"]) for slot in iter_slots(document)]
