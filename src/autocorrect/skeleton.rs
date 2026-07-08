//! Consonant-skeleton matching for the missing-vowel autocorrect channel.
//!
//! A skeleton is the ordered sequence of a word's base consonants with all vowels,
//! vowel-signs, hasant and marks removed. Two words with the same skeleton are candidates
//! for one another when vowels have been dropped (baseline `ক্রল্ম` and its intended
//! spelling `করলাম` both fold to the same key). The corpus is the sole validator.
//!
//! ## No second index — we query the lexicon fst directly
//!
//! `bn.fst` already stores every word and its frequency. Rather than duplicate that into a
//! separate `skeleton → word` index (megabytes of redundant word/frequency bytes), we read
//! skeleton-mates straight out of `bn.fst` with a custom [`fst::Automaton`]
//! ([`SkeletonAutomaton`]) — the same mechanism the Levenshtein corrector already uses. Zero
//! extra storage, full lexicon coverage, and it can never drift out of sync with `bn.fst`.
//!
//! ## What is folded, and why it is NOT the varga grid
//!
//! The traditional varga grid is *Sanskrit* phonology and does not describe *modern
//! Bengali* sound similarity, so we do NOT derive folding from it. Instead the fold set is
//! grounded in the documented grapheme→IPA mapping (Wikipedia *Help:IPA/Bengali* and
//! *Bengali phonology*): we fold only letters that share an IDENTICAL phoneme:
//!   - শ/ষ  → [ʃ]
//!   - জ/য  → [dʒ]
//!   - ণ/ন  → [n]
//!   - ড়/ঢ় → [ɽ]
//!
//! Everything else is kept DISTINCT — including pairs that are merely *nearby*:
//!   - স [s] ≠ শ/ষ [ʃ]   (adjacent fricatives — a graded phonetic confusion, not identical)
//!   - র [ɾ] ≠ ড়/ঢ় [ɽ]   (tap vs retroflex flap)
//!   - ট [ʈ] ≠ ত [t], ক [k] ≠ খ [kʰ]  (place / aspiration are phonemic)
//!
//! Those nearby-phoneme confusions (স↔শ, র↔ড়) and the `t↔T`/`k↔kh` *typing* confusions
//! belong to graded phonetic and keyboard channels — with costs from feature distance —
//! not to a hard skeleton fold.

use fst::automaton::Automaton;

const NUKTA: char = '\u{09BC}';

/// Class for a base consonant. `nukta` is true when the following combining nukta
/// (U+09BC) was seen, forming ড়/ঢ়/য় in NFC-decomposed text; the precomposed forms
/// (U+09DC/U+09DD/U+09DF) are matched directly.
fn consonant_class(ch: char, nukta: bool) -> Option<u8> {
    if nukta {
        return Some(match ch {
            'ড' => b'R', // ড় [ɽ]
            'ঢ' => b'R', // ঢ় [ɽ]
            'য' => b'y', // য়
            _ => return consonant_class(ch, false),
        });
    }
    Some(match ch {
        // velar
        'ক' => b'k',
        'খ' => b'K',
        'গ' => b'g',
        'ঘ' => b'G',
        'ঙ' => b'Y',
        // palatal
        'চ' => b'c',
        'ছ' => b'C',
        'জ' | 'য' => b'j', // homophone fold: জ/য both [dʒ]
        'ঝ' => b'J',
        'ঞ' => b'V',
        // retroflex (place kept distinct from dental)
        'ট' => b'T',
        'ঠ' => b'U',
        'ড' => b'D',
        'ঢ' => b'E',
        // dental
        'ত' | 'ৎ' => b't', // ৎ (khanda-ta) is [t̪] = ত
        'থ' => b'W',
        'দ' => b'd',
        'ধ' => b'F',
        // dental/retroflex nasal merged in modern Bengali
        'ণ' | 'ন' => b'n',
        // labial
        'প' => b'p',
        'ফ' => b'P',
        'ব' => b'b',
        'ভ' => b'B',
        'ম' => b'm',
        // liquids: র [ɾ] and ড়/ঢ় [ɽ] are DISTINCT phonemes (nearby, not identical —
        // the র↔ড় confusion is handled by the graded phonetic channel, not folded here).
        'র' => b'r',
        '\u{09DC}' | '\u{09DD}' => b'R', // precomposed ড়/ঢ়
        'ল' => b'l',
        // sibilants: শ/ষ are both [ʃ] (fold); স is [s], a distinct phoneme.
        'শ' | 'ষ' => b's',
        'স' => b'S',
        'হ' => b'h',
        '\u{09DF}' => b'y', // precomposed য়
        _ => return None,
    })
}

/// Skeletons of one consonant collide with too much of the lexicon; overlong ones are
/// words where a dropped vowel is not the dominant error.
pub const MIN_SKELETON_LEN: usize = 2;
pub const MAX_SKELETON_LEN: usize = 12;

/// Compute the consonant skeleton of a Bengali word. Empty for pure-vowel words.
pub fn consonant_skeleton(word: &str) -> String {
    let mut skeleton = Vec::with_capacity(word.len() / 3 + 1);
    let mut chars = word.chars().peekable();
    while let Some(ch) = chars.next() {
        let nukta = chars.peek() == Some(&NUKTA);
        if nukta {
            chars.next(); // consume the combining nukta
        }
        if let Some(class) = consonant_class(ch, nukta) {
            skeleton.push(class);
        }
    }
    String::from_utf8(skeleton).unwrap_or_default()
}

/// Whether a skeleton is in the indexable/lookup length band.
pub fn is_indexable_skeleton(skeleton: &str) -> bool {
    (MIN_SKELETON_LEN..=MAX_SKELETON_LEN).contains(&skeleton.len())
}

/// A lexicon word that shares the query's consonant skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkeletonMatch {
    pub word: String,
    pub frequency: u64,
}

/// An [`fst::Automaton`] accepting exactly the lexicon words whose consonant skeleton equals
/// `target`. It reads skeleton-mates straight from the main lexicon fst — no separate index.
/// Vowels, vowel-signs, hasant and other marks are skipped; a combining nukta folds into the
/// consonant it follows, mirroring [`consonant_skeleton`] byte-for-byte.
///
/// Matching is *exact*: the word's full consonant sequence must equal `target` (a word with
/// an extra trailing consonant, e.g. skeleton `krn` for query `kr`, is rejected).
pub struct SkeletonAutomaton {
    /// Consonant class bytes — the alphabet emitted by `consonant_class`.
    target: Vec<u8>,
}

impl SkeletonAutomaton {
    /// Build from a baseline word. Returns `None` when the baseline's skeleton is not in the
    /// indexable length band, so the caller can skip the channel entirely.
    pub fn for_baseline(baseline: &str) -> Option<Self> {
        let skeleton = consonant_skeleton(baseline);
        if !is_indexable_skeleton(&skeleton) {
            return None;
        }
        Some(Self {
            target: skeleton.into_bytes(),
        })
    }

    /// Commit a held consonant `ch` (optionally folded with a following nukta): advance if it
    /// matches the next expected class, otherwise kill the branch.
    fn commit(&self, mut state: SkeletonState, ch: char, nukta: bool) -> SkeletonState {
        match consonant_class(ch, nukta) {
            Some(class) => {
                if state.matched < self.target.len() && class == self.target[state.matched] {
                    state.matched += 1;
                    state.held = None;
                } else {
                    state.dead = true;
                }
            }
            // Defensive: a held char that is not a consonant class (should not happen, since
            // we only hold consonants) is treated as a no-op skip.
            None => state.held = None,
        }
        state
    }

    /// Advance the match on a fully decoded character.
    fn on_char(&self, mut state: SkeletonState, ch: char) -> SkeletonState {
        if ch == NUKTA {
            // Fold into the held consonant, if any; a floating nukta is ignored.
            if let Some(held) = state.held.take() {
                return self.commit(state, held, true);
            }
            return state;
        }
        // A non-nukta char first commits any held consonant (no nukta followed it)...
        if let Some(held) = state.held.take() {
            state = self.commit(state, held, false);
            if state.dead {
                return state;
            }
        }
        // ...then either holds this consonant (a nukta may still follow) or skips a
        // vowel / vowel-sign / hasant / other mark.
        if consonant_class(ch, false).is_some() {
            state.held = Some(ch);
        }
        state
    }

    /// The class the held consonant would commit to at word end (no trailing nukta), if any.
    fn trailing_matched(&self, state: &SkeletonState) -> Option<usize> {
        match state.held {
            None => Some(state.matched),
            Some(held) => match consonant_class(held, false) {
                Some(class)
                    if state.matched < self.target.len()
                        && class == self.target[state.matched] =>
                {
                    Some(state.matched + 1)
                }
                Some(_) => None, // held trailing consonant does not match → not a word match
                None => Some(state.matched),
            },
        }
    }
}

/// Incremental UTF-8 + skeleton-match state for one path through the lexicon fst.
#[derive(Clone)]
pub struct SkeletonState {
    /// Consonant classes committed so far (index into `target`).
    matched: usize,
    /// A consonant awaiting a possible following nukta before it is committed.
    held: Option<char>,
    /// Partial UTF-8 bytes of the character currently being decoded.
    pending: [u8; 4],
    pending_len: u8,
    expected_len: u8,
    dead: bool,
}

impl SkeletonState {
    fn dead(&self) -> Self {
        let mut next = self.clone();
        next.dead = true;
        next
    }
}

impl Automaton for SkeletonAutomaton {
    type State = SkeletonState;

    fn start(&self) -> Self::State {
        SkeletonState {
            matched: 0,
            held: None,
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: false,
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        if state.dead || state.pending_len != 0 {
            return false;
        }
        self.trailing_matched(state) == Some(self.target.len())
    }

    fn can_match(&self, state: &Self::State) -> bool {
        !state.dead
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        if state.dead {
            return state.clone();
        }

        if state.pending_len == 0 {
            if byte <= 0x7f {
                // A stray ASCII byte is not part of a Bangla grapheme: treat as a skip
                // (after committing any held consonant).
                return self.on_char(state.clone(), byte as char);
            }
            let Some(expected_len) = utf8_expected_len(byte) else {
                return state.dead();
            };
            let mut next = state.clone();
            next.pending = [0; 4];
            next.pending[0] = byte;
            next.pending_len = 1;
            next.expected_len = expected_len;
            return next;
        }

        if !is_utf8_continuation(byte) {
            return state.dead();
        }

        let mut next = state.clone();
        next.pending[next.pending_len as usize] = byte;
        next.pending_len += 1;
        if next.pending_len < next.expected_len {
            return next;
        }

        let bytes = &next.pending[..next.pending_len as usize];
        let Ok(decoded) = std::str::from_utf8(bytes) else {
            return state.dead();
        };
        let Some(ch) = decoded.chars().next() else {
            return state.dead();
        };
        next.pending = [0; 4];
        next.pending_len = 0;
        next.expected_len = 0;
        self.on_char(next, ch)
    }
}

fn utf8_expected_len(byte: u8) -> Option<u8> {
    match byte {
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

fn is_utf8_continuation(byte: u8) -> bool {
    matches!(byte, 0x80..=0xbf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_and_target_share_a_skeleton() {
        // A vowel-dropped baseline folds to the intended word's skeleton.
        assert_eq!(consonant_skeleton("ক্রল্ম"), consonant_skeleton("করলাম"));
        assert_eq!(consonant_skeleton("দখলাম"), consonant_skeleton("দেখলাম"));
        assert_eq!(consonant_skeleton("ত্মক"), consonant_skeleton("তোমাকে"));
    }

    #[test]
    fn folds_only_identical_phonemes() {
        // Identical-phoneme folds (so মানুষ/মানুশ share a skeleton):
        assert_eq!(consonant_skeleton("শ"), consonant_skeleton("ষ")); // both [ʃ]
        assert_eq!(consonant_skeleton("জ"), consonant_skeleton("য")); // both [dʒ]
        assert_eq!(consonant_skeleton("ণ"), consonant_skeleton("ন")); // both [n]
        assert_eq!(consonant_skeleton("ড়"), consonant_skeleton("ঢ়")); // both [ɽ]
        assert_eq!(consonant_skeleton("মানুষ"), consonant_skeleton("মানুশ"));
    }

    #[test]
    fn nearby_and_distinct_phonemes_stay_distinct() {
        // Nearby but NOT identical — a graded phonetic/keyboard channel handles these:
        assert_ne!(consonant_skeleton("স"), consonant_skeleton("শ")); // [s] vs [ʃ]
        assert_ne!(consonant_skeleton("র"), consonant_skeleton("ড়")); // [ɾ] vs [ɽ]
        assert_ne!(consonant_skeleton("ট"), consonant_skeleton("ত")); // retroflex vs dental
        assert_ne!(consonant_skeleton("ড"), consonant_skeleton("দ"));
        assert_ne!(consonant_skeleton("ক"), consonant_skeleton("খ")); // aspiration is phonemic
        assert_ne!(consonant_skeleton("প"), consonant_skeleton("ফ"));
        assert_ne!(consonant_skeleton("জ"), consonant_skeleton("ঝ")); // জ folds with য, not ঝ
    }

    #[test]
    fn vowels_signs_hasant_and_marks_are_transparent() {
        assert_eq!(consonant_skeleton("আওই"), "");
        assert_eq!(consonant_skeleton("কাজ"), consonant_skeleton("কজ"));
    }

    #[test]
    fn length_band_excludes_singletons_and_overlong() {
        assert!(!is_indexable_skeleton("k"));
        assert!(is_indexable_skeleton("kj"));
    }

    // ---- automaton: it must accept exactly the words with the query skeleton ----

    /// Reference: does `word` have exactly `baseline`'s skeleton (what the automaton must do)?
    fn shares_skeleton(baseline: &str, word: &str) -> bool {
        consonant_skeleton(baseline) == consonant_skeleton(word)
            && is_indexable_skeleton(&consonant_skeleton(baseline))
    }

    /// Drive the automaton over a word's UTF-8 bytes exactly as the fst walk would.
    fn automaton_accepts(baseline: &str, word: &str) -> bool {
        let Some(automaton) = SkeletonAutomaton::for_baseline(baseline) else {
            return false;
        };
        let mut state = automaton.start();
        for &byte in word.as_bytes() {
            if !automaton.can_match(&state) {
                return false;
            }
            state = automaton.accept(&state, byte);
        }
        automaton.is_match(&state)
    }

    #[test]
    fn automaton_matches_skeleton_mates_and_rejects_others() {
        // Cross-check the automaton against the reference skeleton equality on real words.
        let cases = [
            ("ক্রল্ম", "করলাম", true),  // vowel-dropped baseline → intended word
            ("করলাম", "করলাম", true),   // itself
            ("দখলাম", "দেখলাম", true),
            ("মানুশ", "মানুষ", true),    // শ/ষ homophone fold
            ("কর", "করা", true),         // trailing vowel is transparent
            ("কর", "করান", false),       // extra consonant ন → longer skeleton, rejected
            ("কর", "কম", false),         // second consonant differs
            ("কর", "আকর", true),         // leading vowel is transparent
        ];
        for (baseline, word, expected) in cases {
            assert_eq!(
                automaton_accepts(baseline, word),
                expected,
                "automaton({baseline}, {word}) should be {expected}"
            );
            // The automaton must agree with the reference definition it implements.
            if is_indexable_skeleton(&consonant_skeleton(baseline)) {
                assert_eq!(
                    automaton_accepts(baseline, word),
                    shares_skeleton(baseline, word),
                    "automaton disagrees with skeleton equality for ({baseline}, {word})"
                );
            }
        }
    }

    #[test]
    fn automaton_folds_nukta_like_the_skeleton() {
        // বড় (ব + ড + nukta) and its vowelled forms share skeleton bR.
        assert!(automaton_accepts("বড়", "বড়"));
        assert!(automaton_accepts("বড়", "বড়ি"));
        // র (rhotic) must NOT match ড় (flap): distinct phonemes.
        assert!(!automaton_accepts("বর", "বড়"));
    }
}
