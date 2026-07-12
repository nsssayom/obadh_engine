/*
 * obadh.h — stable C ABI for the Obadh transliteration engine.
 *
 * Built when the crate is compiled with the `cabi` feature (native targets only).
 * Link the resulting cdylib/staticlib and include this header.
 *
 * Conventions
 * -----------
 * Sizing (snprintf-style): every function that writes bytes takes an output
 *   pointer and capacity and returns the number of bytes the result needs. It
 *   copies only when the buffer is large enough. Pass a small stack scratch;
 *   reallocate and call again only if the return exceeds the capacity.
 *
 * String lists (count + length-prefixed records, no delimiter): a list is packed
 *   into one buffer as little-endian [uint32 count] followed by `count` records
 *   of [uint32 byte_len][utf8 bytes]. No in-band separator — a candidate may
 *   contain any bytes, and an empty string is faithful.
 *
 * Handles: opaque pointers from *_open / *_new, released by the matching *_free.
 *   Do not use a handle after freeing it, and do not use one handle from multiple
 *   threads at once (no internal locking; the caller owns synchronization).
 *
 * UTF-8: inputs are (ptr, len) UTF-8 spans. Invalid UTF-8 makes the call a no-op
 *   (returns 0 / false / null). A zero length is the empty string.
 *
 * The ABI version (obadh_abi_version) is independent of the crate's semver:
 * additive symbols do not bump it.
 */
#ifndef OBADH_H
#define OBADH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OBADH_ABI_VERSION 1

/* Opaque handles. */
typedef struct ObadhEngine ObadhEngine;
typedef struct ObadhAutocorrect ObadhAutocorrect;
typedef struct ObadhAutosuggest ObadhAutosuggest;

/* Commit strength codes for obadh_autosuggest_commit. */
#define OBADH_COMMIT_ORDINARY            0u
#define OBADH_COMMIT_CORRECTION_REJECTED 1u
#define OBADH_COMMIT_MANUALLY_ADDED      2u

/* ----------------------------------------------------------------- version */

uint32_t obadh_abi_version(void);
size_t   obadh_engine_version(uint8_t *out, size_t cap);

/* ------------------------------------------------------ deterministic engine */

ObadhEngine *obadh_engine_new(void);
void         obadh_engine_free(ObadhEngine *engine);

size_t obadh_transliterate(const ObadhEngine *engine,
                           const uint8_t *input, size_t input_len,
                           uint8_t *out, size_t cap);
size_t obadh_transliterate_lenient(const ObadhEngine *engine,
                                   const uint8_t *input, size_t input_len,
                                   uint8_t *out, size_t cap);

/* ----------------------------------------------------------------- autocorrect */

/* Open from a bn.fst path and an optional loanword FST path (length 0 to omit).
 * Returns NULL on failure. */
ObadhAutocorrect *obadh_autocorrect_open(const uint8_t *fst_path, size_t fst_path_len,
                                         const uint8_t *loanword_path, size_t loanword_path_len);
void              obadh_autocorrect_free(ObadhAutocorrect *autocorrect);

/* Content fingerprint of the loaded lexicon FST (crate <-> artifact check). */
uint64_t obadh_autocorrect_fingerprint(const ObadhAutocorrect *autocorrect);

/* 1 if `word` is an exact lexicon entry, else 0. A real word is never corrected. */
int32_t obadh_autocorrect_is_lexicon_word(const ObadhAutocorrect *autocorrect,
                                          const uint8_t *word, size_t word_len);

/* Ranked correction candidates for `roman` as a packed string list (see header
 * notes). `limit` caps the count. snprintf-style. */
size_t obadh_autocorrect_suggest(const ObadhAutocorrect *autocorrect,
                                 const uint8_t *roman, size_t roman_len,
                                 size_t limit, uint8_t *out, size_t cap);

/* Active-typing candidate bar for `roman`: the deterministic baseline first,
 * then corrections. The baseline is always present so the user can keep what
 * they typed even when it is not a lexicon word. Packed string list. */
size_t obadh_compose_suggestions(const ObadhAutocorrect *autocorrect,
                                 const uint8_t *roman, size_t roman_len,
                                 size_t limit, uint8_t *out, size_t cap);

/* Alternative spellings for an already-composed Bengali `word` (a re-correction
 * menu for a committed word). Input is Bengali. Packed string list. */
size_t obadh_autocorrect_word_alternatives(const ObadhAutocorrect *autocorrect,
                                           const uint8_t *word, size_t word_len,
                                           size_t limit, uint8_t *out, size_t cap);

/* Auto-insert gate. Returns 1 if a correction is confident enough to apply
 * without asking, else 0. When 1, the replacement text is written snprintf-style
 * to out/cap and its needed length to *needed_len; when 0, *needed_len is 0. */
int32_t obadh_autocorrect_should_replace(const ObadhAutocorrect *autocorrect,
                                         const uint8_t *roman, size_t roman_len,
                                         uint8_t *out, size_t cap, size_t *needed_len);

/* ----------------------------------------------------------------- autosuggest */

/* Open from an n-gram artifact path. Returns NULL on failure. */
ObadhAutosuggest *obadh_autosuggest_open(const uint8_t *path, size_t path_len);
void              obadh_autosuggest_free(ObadhAutosuggest *autosuggest);

/* Content fingerprint of the loaded n-gram artifact. */
uint64_t obadh_autosuggest_fingerprint(const ObadhAutosuggest *autosuggest);

/* Commit a token, learning it into the personal overlay. `strength` is one of the
 * OBADH_COMMIT_* codes. Returns 1 if learned, else 0. */
int32_t obadh_autosuggest_commit(ObadhAutosuggest *autosuggest,
                                 const uint8_t *token, size_t token_len, uint32_t strength);

/* Post-decay evidence that the user established `word` (0 if never committed). */
uint32_t obadh_autosuggest_established_weight(const ObadhAutosuggest *autosuggest,
                                              const uint8_t *word, size_t word_len);

/* 1 if the user established `word` with at least `min_weight` post-decay evidence,
 * else 0. Gate for protecting learned words from auto-correction. */
int32_t obadh_autosuggest_is_word_established(const ObadhAutosuggest *autosuggest,
                                              const uint8_t *word, size_t word_len,
                                              uint32_t min_weight);

/* Next-word suggestions for the current session context as a packed string list.
 * Merges the personal overlay's learned words with the model's. */
size_t obadh_autosuggest_suggest(ObadhAutosuggest *autosuggest, size_t limit,
                                 uint8_t *out, size_t cap);

/* Stateless next-word suggestions for an explicit Bengali `context` string.
 * Model-only — does not use or update the session's learned state. */
size_t obadh_autosuggest_suggest_for_context(const ObadhAutosuggest *autosuggest,
                                             const uint8_t *context, size_t context_len,
                                             size_t limit, uint8_t *out, size_t cap);

void obadh_autosuggest_clear_session(ObadhAutosuggest *autosuggest);
void obadh_autosuggest_clear_personal(ObadhAutosuggest *autosuggest);

/* Export/import the personal overlay snapshot. Export is snprintf-style; import
 * returns 1 on success, else 0. */
size_t  obadh_autosuggest_export_personal(const ObadhAutosuggest *autosuggest,
                                          uint8_t *out, size_t cap);
int32_t obadh_autosuggest_import_personal(ObadhAutosuggest *autosuggest,
                                          const uint8_t *input, size_t input_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBADH_H */
