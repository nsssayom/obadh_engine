//! QWERTY key-adjacency ("fat-finger") model for the roman input layer.
//!
//! Obadh is a phonetic IME: the user types a word's sound in roman, which is then
//! transliterated. A *fat-finger* error is a roman-level slip — the finger lands on a
//! physically adjacent key (`bangla` → `banhla`, g→h). Correcting it means rewriting the
//! roman with one adjacent-key substitution, transliterating, and keeping rewrites that yield
//! a real word (validated by the caller against the lexicon).
//!
//! ## Grounding: physical layout + measured touch physics, not a guessed confusion list
//!
//! Two grounded sources set the model, analogous to how the phoneme channel grounds cost in
//! IPA feature distance:
//!
//! 1. **Layout.** Each letter key has an (x, y) coordinate on the standard three-row
//!    staggered QWERTY grid, so which keys physically touch — and how far apart their centres
//!    are — falls out of the coordinates, with no hand-picked confusion pairs.
//!
//! 2. **Touch distribution (FFitts law — Bi, Li & Zhai, CHI 2013).** A finger aimed at a key
//!    lands as a **2-D Gaussian** around its centre. FFitts's dual-distribution model
//!    measured the finger's *absolute* precision at **σ_a ≈ 1.5 mm on a 5 mm-key soft
//!    keyboard ≈ 0.3 key-widths** of intrinsic jitter (independent of speed/target size).
//!    Since a slip's likelihood is that Gaussian evaluated at the neighbour's centre, the
//!    slip cost is **−log P ∝ (key distance)² / 2σ²** — *quadratic* in distance, so near keys
//!    are far cheaper than the linear distance alone would suggest and far keys are sharply
//!    penalised. That is the grounded grading; nothing per-pair is tuned.
//!
//! Case is treated as shift-state: a slip keeps the shift held, so an uppercase key slips to
//! an uppercase neighbour. Adjacency is computed on the physical (lowercase) key and the
//! original character's case is carried through.
//!
//! Known refinement (not modelled): touch endpoints on phones carry a small systematic
//! *downward* offset (users contact below the visual target), which would make down-row slips
//! marginally likelier than up-row. The symmetric Gaussian is the first-order model.

/// Row stagger of a standard ANSI QWERTY, in key-widths: each row sits ~1/2 key right of the
/// one above it, so a key physically sits between — and touches — the two diagonal keys in
/// the adjacent row (g is under both t and y). This half-key stagger reproduces that.
const HOME_ROW_STAGGER: f32 = 0.5;
const BOTTOM_ROW_STAGGER: f32 = 1.0;

/// Effective finger imprecision σ during typing, in key-widths. FFitts law measured the
/// absolute finger precision at σ_a ≈ 1.5 mm on a 5 mm-key soft keyboard (≈ 0.3 key-widths);
/// fast, unverified typing spreads a little wider, so we use ≈ 0.5 key-widths as the working
/// σ. It sets both the neighbour cutoff and the quadratic cost below.
const TOUCH_SIGMA_KEYS: f32 = 0.5;

/// Only keys whose centres are within this many key-widths (≈ 2.4σ) are treated as fat-finger
/// neighbours. ~1.2 admits the immediate horizontal (1.0) and diagonal (~1.12) neighbours on
/// the staggered grid and excludes anything a finger would not plausibly hit in one slip (the
/// next key over is ~2.0 away ≈ 4σ, where the touch Gaussian is negligible).
const NEIGHBOUR_RADIUS: f32 = 1.2;

/// (x, y) centre of each lowercase letter key on the staggered grid.
fn key_coord(ch: char) -> Option<(f32, f32)> {
    // Column index of each key within its row.
    const TOP: &[u8] = b"qwertyuiop";
    const HOME: &[u8] = b"asdfghjkl";
    const BOTTOM: &[u8] = b"zxcvbnm";

    let lower = ch.to_ascii_lowercase() as u8;
    if let Some(col) = TOP.iter().position(|&k| k == lower) {
        return Some((col as f32, 0.0));
    }
    if let Some(col) = HOME.iter().position(|&k| k == lower) {
        return Some((col as f32 + HOME_ROW_STAGGER, 1.0));
    }
    if let Some(col) = BOTTOM.iter().position(|&k| k == lower) {
        return Some((col as f32 + BOTTOM_ROW_STAGGER, 2.0));
    }
    None
}

/// Euclidean distance between two letter keys in key-widths, ignoring case. `None` if either
/// character is not a letter key.
fn key_distance(a: char, b: char) -> Option<f32> {
    let (ax, ay) = key_coord(a)?;
    let (bx, by) = key_coord(b)?;
    Some(((ax - bx).powi(2) + (ay - by).powi(2)).sqrt())
}

/// Physical fat-finger neighbours of `ch` (a different key within [`NEIGHBOUR_RADIUS`]), each
/// paired with its integer slip cost, nearest first. Case is carried through: an uppercase
/// input yields uppercase neighbours (a slip with shift still held).
fn neighbours(ch: char) -> Vec<(char, u16)> {
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let Some((cx, cy)) = key_coord(ch) else {
        return Vec::new();
    };
    let uppercase = ch.is_ascii_uppercase();

    let mut near = Vec::new();
    for &letter in LETTERS {
        let candidate = letter as char;
        if candidate == ch.to_ascii_lowercase() {
            continue;
        }
        let (nx, ny) = key_coord(candidate).expect("letters have coordinates");
        let squared_distance = (cx - nx).powi(2) + (cy - ny).powi(2);
        let distance = squared_distance.sqrt();
        if distance <= NEIGHBOUR_RADIUS {
            let out = if uppercase {
                candidate.to_ascii_uppercase()
            } else {
                candidate
            };
            // FFitts: −log P(slip) ∝ d² / 2σ². Quadratic in distance, so a diagonal slip
            // costs strictly more than a same-row one; rounded to the integer repair scale.
            let cost = (squared_distance / (2.0 * TOUCH_SIGMA_KEYS * TOUCH_SIGMA_KEYS))
                .round()
                .max(1.0) as u16;
            near.push((out, cost, distance));
        }
    }
    near.sort_by(|a, b| a.2.total_cmp(&b.2));
    near.into_iter().map(|(ch, cost, _)| (ch, cost)).collect()
}

/// A roman rewrite formed by one QWERTY key-slip substitution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySlipVariant {
    pub text: String,
    pub cost: u16,
}

/// All single adjacent-key-substitution rewrites of `input`, cheapest first, de-duplicated,
/// up to `max_variants`. Only ASCII-letter positions are perturbed (separators, digits and
/// already-non-letter signals are left alone). The caller transliterates each and keeps only
/// those that yield a real lexicon word — so this stays bounded in practice.
pub fn key_slip_variants(input: &str, max_variants: usize) -> Vec<KeySlipVariant> {
    if max_variants == 0 || input.is_empty() {
        return Vec::new();
    }

    // (exact key distance, text, integer cost). We sort by the exact distance so the nearest
    // slips (horizontal same-row keys, the classic fat-finger) survive the cap ahead of the
    // slightly farther diagonals — the integer cost alone is too coarse to order them.
    let mut scored: Vec<(f32, String, u16)> = Vec::new();
    for (byte_index, ch) in input.char_indices() {
        if !ch.is_ascii_alphabetic() {
            continue;
        }
        for (near, cost) in neighbours(ch) {
            let mut text = String::with_capacity(input.len());
            text.push_str(&input[..byte_index]);
            text.push(near);
            text.push_str(&input[byte_index + ch.len_utf8()..]);
            if text == input || scored.iter().any(|(_, existing, _)| *existing == text) {
                continue;
            }
            let distance = key_distance(ch, near).unwrap_or(f32::MAX);
            scored.push((distance, text, cost));
        }
    }

    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(max_variants);
    scored
        .into_iter()
        .map(|(_, text, cost)| KeySlipVariant { text, cost })
        .collect()
}

/// Cap on key-slip variants transliterated per keystroke. Baked into the engine — not a
/// client tunable. 64 covers a long word's single-slip neighbourhood, and variants are
/// generated nearest-key-first so the cap only ever drops the least likely slips.
const MAX_KEY_SLIP_VARIANTS: usize = 64;

/// Key-slip repaired baselines: single adjacent-key rewrites of `input` whose transliteration
/// is a real lexicon word and differs from the untouched baseline. Mirrors
/// [`super::roman_repaired_outputs`] — the caller supplies only plumbing closures
/// (`transliterate`, `is_lexicon_word`); every numeric parameter is baked into this module.
///
/// **Precision gate:** fires only when the untouched baseline is *not* itself a lexicon word
/// (`baseline_frequency` is `None`), so a validly-typed word is never second-guessed by the
/// fat-finger channel.
pub fn key_slip_repaired_outputs<T, W>(
    input: &str,
    baseline_output: &str,
    baseline_frequency: Option<u64>,
    mut transliterate: T,
    mut is_lexicon_word: W,
) -> Vec<super::roman_repair::RomanRepairedOutput>
where
    T: FnMut(&str) -> String,
    W: FnMut(&str) -> bool,
{
    if baseline_frequency.is_some() {
        return Vec::new();
    }

    let mut outputs = Vec::new();
    for variant in key_slip_variants(input, MAX_KEY_SLIP_VARIANTS) {
        let bangla = transliterate(&variant.text);
        if bangla == baseline_output || !is_lexicon_word(&bangla) {
            continue;
        }
        outputs.push(super::roman_repair::RomanRepairedOutput {
            roman_input: variant.text,
            bangla_output: bangla,
            repair_kind: "qwerty_key_slip",
            repair_cost: variant.cost,
        });
    }
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_are_physically_adjacent_keys() {
        // g sits in the home row between f and h, under t/y, over v/b.
        let near: Vec<char> = neighbours('g').into_iter().map(|(ch, _)| ch).collect();
        for expected in ['f', 'h', 't', 'y', 'v', 'b'] {
            assert!(near.contains(&expected), "g should neighbour {expected}: {near:?}");
        }
        // Far keys are never neighbours.
        for far in ['a', 'p', 'q', 'm', 'l'] {
            assert!(!near.contains(&far), "g should not neighbour {far}");
        }
    }

    #[test]
    fn horizontal_neighbours_are_cheaper_than_diagonal() {
        // f–g share a row (distance 1.0); g–t is diagonal (slightly farther).
        let f_g = key_distance('f', 'g').unwrap();
        let g_t = key_distance('g', 't').unwrap();
        assert!(f_g < g_t, "same-row slip should be nearer than diagonal");
        assert!((f_g - 1.0).abs() < 1e-6);
    }

    #[test]
    fn case_is_carried_through_as_shift_state() {
        let near = neighbours('S');
        assert!(near.iter().all(|(ch, _)| ch.is_ascii_uppercase()));
        // S neighbours A/D/W/X/E/Z on the physical layout, all shifted.
        let chars: Vec<char> = near.iter().map(|(ch, _)| *ch).collect();
        assert!(chars.contains(&'A') && chars.contains(&'D'));
    }

    #[test]
    fn variants_are_single_substitutions_bounded_and_ordered() {
        let variants = key_slip_variants("bangla", 12);
        assert!(!variants.is_empty());
        // Each differs from the input in exactly one character position.
        for variant in &variants {
            assert_eq!(variant.text.chars().count(), "bangla".chars().count());
            let diffs = variant
                .text
                .chars()
                .zip("bangla".chars())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(diffs, 1, "{} should differ in one position", variant.text);
        }
        // banhla (g→h) is a plausible adjacent slip and must be generated.
        assert!(variants.iter().any(|variant| variant.text == "banhla"));
        // Bounded and cheapest-first.
        assert!(variants.len() <= 12);
        assert!(variants.windows(2).all(|w| w[0].cost <= w[1].cost));
    }

    #[test]
    fn non_letter_positions_are_left_alone() {
        // Digits/separators are not perturbed; only the letters around them are.
        let variants = key_slip_variants("a5", 8);
        assert!(variants.iter().all(|variant| variant.text.ends_with('5')));
    }

    /// Recall/precision of the key-slip channel over the real lexicon. The "intended word" is
    /// defined as `transliterate(roman)` — never hand-asserted — and only romans that map to a
    /// sufficiently frequent lexicon word are kept, so the trial set is corpus-validated. For
    /// each such roman we inject every single adjacent-key slip and measure how often the full
    /// pipeline recovers the intended word. Ignored by default (needs resolved submodule data):
    ///   cargo test --release --lib key_slip_recall_and_precision_probe -- --ignored --nocapture
    #[test]
    #[ignore = "needs the resolved data/autocorrect/models/bn.fst; QWERTY recall/precision probe"]
    fn key_slip_recall_and_precision_probe() {
        use crate::{
            roman_repaired_outputs, FstLexicon, FstRepairedBaseline, FstSuggestOptions,
            ObadhEngine, RomanRepairOptions, DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
        };

        let Ok(bytes) = std::fs::read("data/autocorrect/models/bn.fst") else {
            eprintln!("skip: bn.fst not resolved");
            return;
        };
        let lexicon = FstLexicon::from_bytes(bytes).expect("load fst");
        let engine = ObadhEngine::new();
        let options = FstSuggestOptions {
            max_distance: 2,
            max_edit_cost: None,
            max_candidates: 512,
            max_prefix_candidates: 8,
            response_candidates: 8,
        };
        const MIN_FREQ: u64 = 500;

        // Common Bengali-word romans; each is used only if it transliterates to a word with
        // frequency >= MIN_FREQ (others are skipped, keeping the set honest and corpus-defined).
        let romans = [
            "ami", "tumi", "amar", "tomar", "bangla", "desh", "bhalo", "kemon", "manush",
            "kotha", "kaj", "din", "raat", "boi", "naam", "ghor", "jol", "gaan", "chokh",
            "hat", "mon", "jibon", "somoy", "bhasha", "chele", "meye", "baba", "bhai", "bon",
            "sokal", "ekhon", "tokhon", "jonno", "karon", "kintu", "ebong", "kore", "kori",
            "bola", "dekha", "jani", "hobe", "ache", "bikel", "shohor", "gram", "rasta",
            "notun", "purano", "boro", "choto",
        ];

        // Full production pipeline for one roman input → ranked candidate texts.
        let run = |roman: &str| -> Vec<String> {
            let base = engine.transliterate(roman);
            let mut reps = roman_repaired_outputs(
                roman,
                &base,
                RomanRepairOptions {
                    max_repairs: DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
                },
                |repair| engine.transliterate(repair),
            );
            reps.extend(key_slip_repaired_outputs(
                roman,
                &base,
                lexicon.exact_frequency(&base),
                |repair| engine.transliterate(repair),
                |word| lexicon.exact_frequency(word).is_some(),
            ));
            let baselines: Vec<FstRepairedBaseline> = reps
                .iter()
                .map(|repair| FstRepairedBaseline {
                    roman_input: &repair.roman_input,
                    bangla_output: &repair.bangla_output,
                    repair_kind: repair.repair_kind,
                    repair_cost: repair.repair_cost,
                })
                .collect();
            lexicon
                .suggest_with_repaired_baselines(&base, &baselines, options)
                .expect("suggest")
                .candidates
                .into_iter()
                .map(|candidate| candidate.text)
                .collect()
        };

        let (mut valid, mut precision_kept) = (0usize, 0usize);
        let (mut trials, mut recall_1, mut recall_5) = (0usize, 0usize, 0usize);
        // "Gate-open" = the slip produced a non-word baseline, which is exactly when the
        // channel is designed to act. Slips that happen to land on another valid word are
        // deliberately left alone (precision), so they are reported separately.
        let (mut gate_open, mut gate_open_r1, mut gate_open_r5) = (0usize, 0usize, 0usize);
        let mut slip_is_word = 0usize;
        for roman in romans {
            let expected = engine.transliterate(roman);
            match lexicon.exact_frequency(&expected) {
                Some(frequency) if frequency >= MIN_FREQ => {}
                _ => continue,
            }
            valid += 1;

            // Precision: the correctly-typed roman must keep the intended word at #1.
            if run(roman).first().map(String::as_str) == Some(expected.as_str()) {
                precision_kept += 1;
            }

            // Recall: inject every single adjacent-key slip and try to recover `expected`.
            for slip in key_slip_variants(roman, 64) {
                let base = engine.transliterate(&slip.text);
                if base == expected {
                    continue; // slip did not change the transliteration → not a test case
                }
                trials += 1;
                let gate_is_open = lexicon.exact_frequency(&base).is_none();
                if !gate_is_open {
                    slip_is_word += 1; // slip hit another valid word; channel stays out by design
                }
                let ranked = run(&slip.text);
                let rank = ranked.iter().position(|text| *text == expected);
                if rank == Some(0) {
                    recall_1 += 1;
                }
                if rank.is_some_and(|position| position < 5) {
                    recall_5 += 1;
                }
                if gate_is_open {
                    gate_open += 1;
                    if rank == Some(0) {
                        gate_open_r1 += 1;
                    }
                    if rank.is_some_and(|position| position < 5) {
                        gate_open_r5 += 1;
                    }
                }
            }
        }

        let pct = |num: usize, den: usize| if den == 0 { 0.0 } else { 100.0 * num as f64 / den as f64 };
        eprintln!(
            "QWERTY precision: {precision_kept}/{valid} correctly-typed words kept at #1 (target 100%)"
        );
        eprintln!(
            "recall over ALL {trials} injected slips:            @1 {:.1}%  @5 {:.1}%",
            pct(recall_1, trials),
            pct(recall_5, trials),
        );
        eprintln!(
            "recall over the {gate_open} gate-OPEN slips (non-word typo): @1 {:.1}%  @5 {:.1}%",
            pct(gate_open_r1, gate_open),
            pct(gate_open_r5, gate_open),
        );
        eprintln!("({slip_is_word} slips landed on another valid word — left untouched by design)");
        assert_eq!(precision_kept, valid, "a correctly-typed word must never be demoted");
    }
}
