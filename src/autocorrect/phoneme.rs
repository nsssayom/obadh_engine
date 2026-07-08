//! Bengali consonant phoneme distances from articulatory features.
//!
//! This is the graded confusion metric behind the autocorrect error model: how likely is
//! one consonant to be written/typed for another. It is NOT binary "same/different" — it
//! returns a distance so that *nearby* phonemes (স [s] ↔ শ [ʃ]) cost little while distant
//! ones (ক [k] ↔ প [p]) cost a lot.
//!
//! ## Grounding (not intuition)
//!
//! Feature values come from the documented Bengali grapheme→IPA inventory (Wikipedia
//! *Help:IPA/Bengali* and *Bengali phonology*). The distance is a weighted articulatory
//! feature difference — the feature-based-Levenshtein approach that correlates with human
//! confusability at r≈0.925 (Fabiano-Smith & Hoffman methodology).
//!
//! Key Bengali facts encoded here:
//! - The "retroflex" series (ট ঠ ড ঢ) is **apical post-alveolar**, and শ/ষ are
//!   post-alveolar [ʃ]; স is alveolar [s]. So the place scale puts alveolar and
//!   post-alveolar **adjacent** — স↔শ is one place step, correctly close.
//! - Aspiration and voicing are phonemic; each is one graded feature.
//! - শ=ষ ([ʃ]), জ=য ([dʒ]), ণ=ন ([n]), ড়=ঢ় ([ɽ]) collapse to one phoneme → distance 0.

// The graded confusion channel that consumes these distances is the next integration step.
#![allow(dead_code)]

/// Articulatory place, ordered front→back so `|place difference|` is itself a graded
/// distance (adjacent places are the most confusable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    Labial = 0,
    Dental = 1,
    Alveolar = 2,
    PostAlveolar = 3, // Bengali "retroflex" (apical) + ʃ live here
    Palatal = 4,      // tʃ/dʒ affricates, j
    Velar = 5,
    Glottal = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Manner {
    Stop,
    Affricate,
    Nasal,
    Fricative,
    Lateral,
    Rhotic,
    Approximant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Phoneme {
    place: Place,
    manner: Manner,
    voiced: bool,
    aspirated: bool, // aspirated OR breathy-voiced
}

// Feature-difference weights. Chosen to rank the common Bengali spelling confusions
// cheapest (aspiration/place-adjacent) and unrelated substitutions expensive; the ORDERING
// they produce is asserted in the tests against known-close / known-far pairs. They are the
// one deliberate modelling parameter here, kept small and interpretable.
const W_PLACE: u16 = 2; // per place step
const W_MANNER: u16 = 3; // per manner step
const W_VOICE: u16 = 2;
const W_ASPIRATION: u16 = 1; // aspiration is the most commonly dropped/added feature

/// Graded manner distance: obstruents (stop/affricate/fricative) are mutually closer than
/// they are to sonorants (nasal/lateral/rhotic/approximant).
fn manner_distance(a: Manner, b: Manner) -> u16 {
    use Manner::*;
    if a == b {
        return 0;
    }
    let obstruent = |m: Manner| matches!(m, Stop | Affricate | Fricative);
    let sonorant = |m: Manner| matches!(m, Nasal | Lateral | Rhotic | Approximant);
    // stop↔affricate and affricate↔fricative are the closest cross-manner steps.
    match (a, b) {
        (Stop, Affricate) | (Affricate, Stop) => 1,
        (Affricate, Fricative) | (Fricative, Affricate) => 1,
        (Stop, Fricative) | (Fricative, Stop) => 2,
        _ if obstruent(a) && obstruent(b) => 2,
        _ if sonorant(a) && sonorant(b) => 2, // e.g. nasal↔lateral
        _ => 4,                               // obstruent↔sonorant
    }
}

fn phoneme_distance(a: Phoneme, b: Phoneme) -> u16 {
    let place = (a.place as i16 - b.place as i16).unsigned_abs() * W_PLACE;
    let manner = manner_distance(a.manner, b.manner) * W_MANNER;
    let voice = u16::from(a.voiced != b.voiced) * W_VOICE;
    let aspiration = u16::from(a.aspirated != b.aspirated) * W_ASPIRATION;
    place + manner + voice + aspiration
}

/// Map a Bengali base consonant (single scalar; handle nukta via [`consonant_distance`])
/// to its phoneme, folding same-IPA graphemes (শ/ষ, জ/য, ণ/ন) to one value.
fn base_phoneme(ch: char) -> Option<Phoneme> {
    use Manner::*;
    use Place::*;
    let p = |place, manner, voiced, aspirated| {
        Some(Phoneme {
            place,
            manner,
            voiced,
            aspirated,
        })
    };
    match ch {
        // velar stops
        'ক' => p(Velar, Stop, false, false),
        'খ' => p(Velar, Stop, false, true),
        'গ' => p(Velar, Stop, true, false),
        'ঘ' => p(Velar, Stop, true, true),
        'ঙ' => p(Velar, Nasal, true, false),
        // palatal affricates
        'চ' => p(Palatal, Affricate, false, false),
        'ছ' => p(Palatal, Affricate, false, true),
        'জ' | 'য' => p(Palatal, Affricate, true, false), // both [dʒ]
        'ঝ' => p(Palatal, Affricate, true, true),
        'ঞ' => p(Palatal, Nasal, true, false),
        // post-alveolar ("retroflex") stops
        'ট' => p(PostAlveolar, Stop, false, false),
        'ঠ' => p(PostAlveolar, Stop, false, true),
        'ড' => p(PostAlveolar, Stop, true, false),
        'ঢ' => p(PostAlveolar, Stop, true, true),
        // dental stops
        'ত' | 'ৎ' => p(Dental, Stop, false, false),
        'থ' => p(Dental, Stop, false, true),
        'দ' => p(Dental, Stop, true, false),
        'ধ' => p(Dental, Stop, true, true),
        // nasals (ণ merged to alveolar [n])
        'ণ' | 'ন' => p(Alveolar, Nasal, true, false),
        // labials
        'প' => p(Labial, Stop, false, false),
        'ফ' => p(Labial, Stop, false, true),
        'ব' => p(Labial, Stop, true, false),
        'ভ' => p(Labial, Stop, true, true),
        'ম' => p(Labial, Nasal, true, false),
        // liquids
        'র' => p(Alveolar, Rhotic, true, false),       // [ɾ]
        '\u{09DC}' | '\u{09DD}' => p(PostAlveolar, Rhotic, true, false), // ড়/ঢ় [ɽ]
        'ল' => p(Alveolar, Lateral, true, false),
        // fricatives
        'শ' | 'ষ' => p(PostAlveolar, Fricative, false, false), // [ʃ]
        'স' => p(Alveolar, Fricative, false, false),           // [s]
        'হ' => p(Glottal, Fricative, true, false),
        '\u{09DF}' => p(Palatal, Approximant, true, false), // য় [j]
        _ => None,
    }
}

/// Distance between two Bengali consonants given as `(char, has_nukta)`. Returns `None`
/// if either is not a mapped consonant. Identical phonemes → 0; nearby → small; far → large.
pub fn consonant_distance(a: (char, bool), b: (char, bool)) -> Option<u16> {
    Some(phoneme_distance(nukta_phoneme(a)?, nukta_phoneme(b)?))
}

/// Single-codepoint Bengali base consonants, for enumerating near-phoneme neighbours.
const BASE_CONSONANTS: &[char] = &[
    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ',
    'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ',
];

/// Base consonants whose phoneme is within `max_distance` of `ch` (a different grapheme),
/// each paired with the distance, nearest first. Distance 0 is included (শ↔ষ, জ↔য, ণ↔ন),
/// so genuine same-sound spelling confusions are enumerated alongside graded near ones.
/// Only single-codepoint consonants are produced (nukta forms are handled by the caller).
pub fn near_consonants(ch: char, max_distance: u16) -> Vec<(char, u16)> {
    let Some(source) = base_phoneme(ch) else {
        return Vec::new();
    };
    let mut near = Vec::new();
    for &other in BASE_CONSONANTS {
        if other == ch {
            continue;
        }
        if let Some(candidate) = base_phoneme(other) {
            let distance = phoneme_distance(source, candidate);
            if distance <= max_distance {
                near.push((other, distance));
            }
        }
    }
    near.sort_by_key(|&(_, distance)| distance);
    near
}

fn nukta_phoneme((ch, nukta): (char, bool)) -> Option<Phoneme> {
    if nukta {
        return match ch {
            'ড' | 'ঢ' => Some(Phoneme {
                place: Place::PostAlveolar,
                manner: Manner::Rhotic,
                voiced: true,
                aspirated: false,
            }), // ড়/ঢ় [ɽ]
            'য' => base_phoneme('\u{09DF}'), // য় [j]
            _ => base_phoneme(ch),
        };
    }
    base_phoneme(ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: char, b: char) -> u16 {
        consonant_distance((a, false), (b, false)).expect("mapped consonants")
    }

    #[test]
    fn identical_phonemes_are_zero() {
        assert_eq!(dist('শ', 'ষ'), 0); // both [ʃ]
        assert_eq!(dist('জ', 'য'), 0); // both [dʒ]
        assert_eq!(dist('ণ', 'ন'), 0); // both [n]
        assert_eq!(dist('ক', 'ক'), 0);
    }

    #[test]
    fn nearby_phonemes_are_small_and_ordered_below_far_ones() {
        // The confusions we care about are all SMALL:
        let s_sh = dist('স', 'শ'); // [s] vs [ʃ]: one place step
        let k_kh = dist('ক', 'খ'); // aspiration only
        let p_ph = dist('প', 'ফ');
        let t_stop = dist('ত', 'ট'); // dental vs post-alveolar stop
        let r_flap = consonant_distance(('র', false), ('ড', true)).unwrap(); // র [ɾ] vs ড় [ɽ]

        // ...and all clearly below UNrelated substitutions:
        let k_p = dist('ক', 'প'); // velar vs labial stop
        let m_s = dist('ম', 'স'); // labial nasal vs alveolar fricative
        let l_k = dist('ল', 'ক');

        for near in [s_sh, k_kh, p_ph, t_stop, r_flap] {
            assert!(near > 0, "nearby but not identical");
            assert!(near < k_p, "nearby {near} should be < far ক/প {k_p}");
            assert!(near < m_s && near < l_k, "nearby {near} should be < unrelated");
        }
        // aspiration is the cheapest single confusion.
        assert!(k_kh <= s_sh && k_kh <= t_stop);
    }

    #[test]
    fn nukta_letters_map_to_flap() {
        // ড় (ড + nukta) is [ɽ], close to র [ɾ], far from ড [ɖ] stop.
        let r_flap = consonant_distance(('র', false), ('ড', true)).unwrap();
        let stop_flap = consonant_distance(('ড', false), ('ড', true)).unwrap();
        assert!(r_flap < stop_flap, "ড় should be closer to র than to ড");
    }
}
