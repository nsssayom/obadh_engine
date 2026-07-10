//! Reference-free properties of the deterministic core.
//!
//! Every assertion here is checkable without knowing the expected Bengali for a
//! given Roman input. That is the point: they hold for inputs nobody tabulated,
//! so they keep guarding the engine as the rule tables grow.
//!
//! The corpus is generated from a fixed seed. It is deliberately not random per
//! run — a property test that finds a different bug on every CI run is a flaky
//! test, not a regression net. When a property fails, reduce the failing input
//! to a minimal reproducer and pin it as an ordinary example test.
//!
//! Two corpora are used. `clean_corpus` stays inside the sanitizer's allowed
//! character set; `dirty_corpus` deliberately leaves it. The distinction
//! matters: strict `transliterate` returns unsupported input unchanged, so
//! properties about rendering only hold on clean input.

use obadh_engine::{ObadhEngine, Sanitizer, TokenType};
use std::sync::OnceLock;

const DOTTED_CIRCLE: char = '\u{25CC}';

const CONSONANTS: &[&str] = &[
    "k", "kh", "g", "gh", "Ng", "c", "ch", "j", "jh", "NG", "T", "Th", "D", "Dh", "N", "t", "th",
    "d", "dh", "n", "p", "ph", "b", "bh", "m", "z", "r", "l", "sh", "Sh", "s", "h", "x", "R",
];

const VOWELS: &[&str] = &["a", "A", "i", "I", "u", "U", "e", "E", "o", "O", "OI", "OU"];

const SIGNALS: &[&str] = &[
    "", ",,", "``", "^", ":", "y", "w", "rr", "ng", "..", ".", "1", "23",
];

/// Characters the sanitizer accepts.
const ALLOWED: &[u8] = b"aAiIuUeEoOkgcjtTdDnpbmyrlshHNzwxRSC.,`^:-_0123456789 ";

/// Characters the sanitizer rejects. Multi-byte on purpose: they also exercise
/// the byte-indexed scanners against landing mid-codepoint.
const DISALLOWED: &[char] = &['é', 'ñ', 'ü', '~', '€', '§', '±', '日', '🎉', '¿'];

/// Fixed-seed LCG. Same corpus on every machine and every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
}

/// Structured combinations, a pseudo-random soup, and the engine's own Bengali
/// output fed back in. Every input here is inside the sanitizer's contract.
fn build_clean_corpus() -> Vec<String> {
    let engine = ObadhEngine::new();
    let mut inputs: Vec<String> = Vec::new();

    for first in CONSONANTS {
        for second in CONSONANTS {
            for vowel in VOWELS {
                inputs.push(format!("{first}{second}{vowel}"));
            }
            for signal in SIGNALS {
                inputs.push(format!("{first}{signal}{second}a"));
            }
        }
    }

    for vowel in VOWELS {
        for signal in SIGNALS {
            for consonant in CONSONANTS {
                inputs.push(format!("{vowel}{signal}{consonant}"));
                inputs.push(format!("{consonant}{vowel}{signal}"));
                inputs.push(format!("{signal}{vowel}{consonant}"));
            }
        }
    }

    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    for _ in 0..20_000 {
        let length = 1 + rng.next() % 14;
        let word: String = (0..length)
            .map(|_| ALLOWED[rng.next() % ALLOWED.len()] as char)
            .collect();
        inputs.push(word);
    }

    // Bengali is inside the allowed set, so the engine's own output is valid
    // input. Feeding it back exercises the pass-through paths.
    let bengali: Vec<String> = inputs
        .iter()
        .take(5_000)
        .map(|input| engine.transliterate(input))
        .collect();
    inputs.extend(bengali);

    assert!(inputs.len() > 40_000, "corpus too small: {}", inputs.len());
    inputs
}

/// Clean inputs with rejected characters spliced in at every position, plus a
/// mixed soup. Without these the `lenient == strict(clean)` property is vacuous:
/// `clean(x)` would equal `x` for every input.
fn build_dirty_corpus() -> Vec<String> {
    let mut rng = Rng(0x13198A2E_03707344);
    let mut inputs: Vec<String> = Vec::new();

    for (index, base) in clean_corpus().iter().take(3_000).enumerate() {
        let bad = DISALLOWED[index % DISALLOWED.len()];
        inputs.push(format!("{bad}{base}"));
        inputs.push(format!("{base}{bad}"));

        let middle = base.len() / 2;
        if base.is_char_boundary(middle) {
            inputs.push(format!("{}{bad}{}", &base[..middle], &base[middle..]));
        }
    }

    for _ in 0..5_000 {
        let length = 1 + rng.next() % 12;
        let word: String = (0..length)
            .map(|_| {
                if rng.next() % 3 == 0 {
                    DISALLOWED[rng.next() % DISALLOWED.len()]
                } else {
                    ALLOWED[rng.next() % ALLOWED.len()] as char
                }
            })
            .collect();
        inputs.push(word);
    }

    assert!(
        inputs.len() > 10_000,
        "dirty corpus too small: {}",
        inputs.len()
    );
    inputs
}

fn clean_corpus() -> &'static [String] {
    static CLEAN: OnceLock<Vec<String>> = OnceLock::new();
    CLEAN.get_or_init(build_clean_corpus)
}

fn dirty_corpus() -> &'static [String] {
    static DIRTY: OnceLock<Vec<String>> = OnceLock::new();
    DIRTY.get_or_init(build_dirty_corpus)
}

fn report(violations: &[String], property: &str, total: usize) {
    assert!(
        violations.is_empty(),
        "{} of {total} inputs violated {property}, first 10:\n{}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Bengali is inside the allowed character set, so the engine's output is itself
/// valid input. Transliterating it again must be a no-op: the output is a fixed
/// point. A violation means some Bengali the engine emits is re-read as Roman on
/// a second pass, which would corrupt already-committed text.
#[test]
fn transliterate_is_idempotent_on_its_own_output() {
    let engine = ObadhEngine::new();
    let mut violations = Vec::new();
    let inputs: Vec<&String> = clean_corpus().iter().chain(dirty_corpus()).collect();

    for input in &inputs {
        let once = engine.transliterate(input);
        let twice = engine.transliterate(&once);
        if twice != once {
            violations.push(format!("{input:?}: {once:?} -> {twice:?}"));
        }
    }

    report(&violations, "idempotence", inputs.len());
}

/// `transliterate` renders straight from text; `transliterate_tokens` renders
/// from a token stream. They are two independent implementations of the same
/// mapping, so on supported input they are each other's oracle.
///
/// Scoped to clean input on purpose — see
/// `strict_text_path_and_token_path_diverge_on_unsupported_input`.
#[test]
fn text_and_token_render_paths_agree_on_supported_input() {
    let engine = ObadhEngine::new();
    let inputs = clean_corpus();
    let mut violations = Vec::new();

    for input in inputs {
        let via_text = engine.transliterate(input);
        let via_tokens = engine.transliterate_tokens(&engine.tokenize(input));
        if via_text != via_tokens {
            violations.push(format!(
                "{input:?}: text={via_text:?} tokens={via_tokens:?}"
            ));
        }
    }

    report(&violations, "text/token render agreement", inputs.len());
}

/// `transliterate_lenient` is defined as dropping unsupported characters and
/// then transliterating, so it must equal the strict path applied to cleaned
/// input. The dirty corpus is what gives this property teeth: on clean input
/// `clean(x) == x` and the assertion is vacuous.
#[test]
fn lenient_equals_strict_on_cleaned_input() {
    let engine = ObadhEngine::new();
    let sanitizer = Sanitizer::new();
    let mut violations = Vec::new();
    let inputs: Vec<&String> = clean_corpus().iter().chain(dirty_corpus()).collect();

    let mut rejected = 0usize;
    for input in &inputs {
        let cleaned = sanitizer.clean(input);
        if cleaned != **input {
            rejected += 1;
        }
        let lenient = engine.transliterate_lenient(input);
        let strict = engine.transliterate(&cleaned);
        if lenient != strict {
            violations.push(format!("{input:?}: lenient={lenient:?} strict={strict:?}"));
        }
    }

    assert!(
        rejected > 10_000,
        "property is vacuous: only {rejected} inputs had a character to clean"
    );
    report(&violations, "lenient == strict(clean)", inputs.len());
}

/// Tokenizing must not lose or invent input. Token contents reassemble the
/// original text exactly, and positions never go backwards.
#[test]
fn tokenizer_spans_reassemble_the_input() {
    let engine = ObadhEngine::new();
    let mut violations = Vec::new();
    let inputs: Vec<&String> = clean_corpus().iter().chain(dirty_corpus()).collect();

    for input in &inputs {
        let tokens = engine.tokenize(input);

        let rebuilt: String = tokens.iter().map(|token| token.content.as_str()).collect();
        if rebuilt != **input {
            violations.push(format!("{input:?}: reassembled as {rebuilt:?}"));
            continue;
        }

        let mut previous = 0usize;
        for token in &tokens {
            if token.position < previous {
                violations.push(format!("{input:?}: token positions go backwards"));
                break;
            }
            previous = token.position;
        }
    }

    report(&violations, "tokenizer span coverage", inputs.len());
}

/// A dotted circle is what a renderer substitutes when a combining mark has no
/// base. The engine must never emit one, for any input, on any path.
#[test]
fn no_render_path_emits_a_dotted_circle() {
    let engine = ObadhEngine::new();
    let mut violations = Vec::new();
    let inputs: Vec<&String> = clean_corpus().iter().chain(dirty_corpus()).collect();

    for input in &inputs {
        for (path, output) in [
            ("text", engine.transliterate(input)),
            ("lenient", engine.transliterate_lenient(input)),
            (
                "tokens",
                engine.transliterate_tokens(&engine.tokenize(input)),
            ),
        ] {
            if output.contains(DOTTED_CIRCLE) {
                violations.push(format!("{input:?} via {path} -> {output:?}"));
            }
        }
    }

    report(&violations, "no dotted circle", inputs.len());
}

/// Strict `transliterate` returns unsupported input unchanged.
#[test]
fn strict_transliterate_passes_unsupported_input_through() {
    let engine = ObadhEngine::new();

    for input in ["héllo", "naïve", "日本語", "emoji 🎉", "k~a"] {
        assert_eq!(engine.transliterate(input), input, "{input}");
    }

    let tokens = engine.tokenize("ami 42 bhalo, achi.");
    let has = |kind: TokenType| tokens.iter().any(|token| token.token_type == kind);
    assert!(has(TokenType::Word));
    assert!(has(TokenType::Number));
    assert!(has(TokenType::Whitespace));
    assert!(has(TokenType::Punctuation));
}

/// The two render paths do **not** agree on unsupported input, and that is not
/// an oversight in these tests.
///
/// `transliterate` is strict: one unsupported character anywhere makes it return
/// the whole input unchanged. `transliterate_tokens` carries no such guard — it
/// renders the supported tokens and passes the rest through. Callers on the
/// token path (the WASM surface, incremental keyboard integrations) therefore
/// see partial transliteration where the text path refuses.
///
/// Pinned so any change to it is a deliberate one. See issue #16.
#[test]
fn strict_text_path_and_token_path_diverge_on_unsupported_input() {
    let engine = ObadhEngine::new();

    for (input, via_text, via_tokens) in [
        ("ké", "ké", "কé"),
        ("k~a", "k~a", "ক~আ"),
        ("日k", "日k", "日ক"),
    ] {
        assert_eq!(engine.transliterate(input), via_text, "text path: {input}");
        assert_eq!(
            engine.transliterate_tokens(&engine.tokenize(input)),
            via_tokens,
            "token path: {input}"
        );
    }
}
