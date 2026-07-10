use obadh_engine::{ObadhEngine, PhoneticUnitType, Tokenizer};

#[test]
fn test_explicit_hasant_notation() {
    let tokenizer = Tokenizer::new();

    // Test explicit hasant notation (n,,d,,r)
    let units = tokenizer.tokenize_word("n,,d,,r");
    // Verify we get a conjunct
    assert!(!units.is_empty());
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);

    // A trailing hasant is a deliberate dead-cluster command, so it must
    // remain visible after the explicit conjunct is formed.
    let units = tokenizer.tokenize_word("n,,d,,r,,");
    assert!(!units.is_empty());
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
    assert_eq!(units[0].text, "n,,d,,r");
    assert_eq!(units[1].unit_type, PhoneticUnitType::ConsonantWithHasant);
    assert_eq!(units[1].text, ",,");

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("n,,d,,r n,,d,,r,,"), "ন্দ্র ন্দ্র্");
}

#[test]
fn test_explicit_hasant_with_vowel() {
    let tokenizer = Tokenizer::new();

    // Test explicit hasant with vowel (n,,d,,rA)
    let units = tokenizer.tokenize_word("n,,d,,rA");
    // Verify we get a conjunct with vowel
    assert!(!units.is_empty());
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithVowel);

    // Test with terminator vowel (n,,d,,ro)
    let units = tokenizer.tokenize_word("n,,d,,ro");
    // Verify we get a conjunct with terminator
    assert!(!units.is_empty());
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithTerminator);
}

#[test]
fn test_mixed_hasant_notation() {
    let tokenizer = Tokenizer::new();

    // Test mixing auto and explicit hasant (k,,lr)
    let units = tokenizer.tokenize_word("k,,lr");
    // Verify the output structure
    assert!(!units.is_empty());

    // Test more complex mixed notation (n,,dr)
    let units = tokenizer.tokenize_word("n,,dr");
    // Verify the output structure
    assert!(!units.is_empty());
}

// ------------------------------------------------------- hasant attachment

const HASANT: char = '\u{09CD}';

const CONSONANTS: &[&str] = &[
    "k", "kh", "g", "gh", "Ng", "c", "ch", "j", "jh", "NG", "T", "Th", "D", "Dh", "N", "t", "th",
    "d", "dh", "n", "p", "ph", "b", "bh", "m", "z", "r", "l", "sh", "Sh", "s", "h", "x", "R",
];

const VOWELS: &[&str] = &[
    "", "a", "A", "i", "I", "u", "U", "e", "E", "o", "O", "OI", "OU",
];

/// Signals that render a sign which cannot carry a hasant.
const MARKERS: &[&str] = &["", "^", ":", "ng", "M", "1", "rr", "t``"];

/// Whether a rendered Bengali character can carry a hasant. Khanda ta (ৎ) is
/// excluded: it is already a dead consonant.
fn bears_hasant(character: char) -> bool {
    matches!(
        character,
        '\u{0995}'..='\u{09B9}' | '\u{09BC}' | '\u{09DC}' | '\u{09DD}' | '\u{09DF}'
    )
}

/// A hasant suppresses a consonant's inherent vowel. Once an explicit vowel sign
/// has been written there is nothing left to suppress, so the `,,` signal is
/// dropped rather than stacked onto a sign that cannot carry it.
#[test]
fn test_hasant_is_dropped_when_it_has_no_consonant_to_sit_on() {
    let engine = ObadhEngine::new();

    for (input, expected) in [
        // after a kar (dependent vowel sign)
        ("ka,,", "কা"),
        ("ki,,", "কি"),
        ("kOI,,", "কৈ"),
        ("ka,,k", "কাক"),
        // the base carrying the kar may itself be a conjunct or a reph
        ("pxa,,", "পক্সা"),
        ("krrka,,", "কর্কা"),
        // after signs the engine already treats as non-joining
        ("k^,,", "কঁ"),
        ("k:,,", "কঃ"),
        ("kng,,", "কং"),
        ("kt``,,", "কৎ"),
        // after a numeral, and after another hasant
        ("k1,,", "ক১"),
        ("rr,,", "র্"),
    ] {
        assert_eq!(engine.transliterate(input), expected, "{input}");
    }
}

/// Everything the rule sources document about `,,` keeps working: the hasant is
/// legitimate whenever it has a consonant to sit on, including a consonant that
/// carries only the inherent vowel.
#[test]
fn test_hasant_still_attaches_to_a_consonant() {
    let engine = ObadhEngine::new();

    for (input, expected) in [
        ("k,,", "ক্"),
        ("ko,,", "ক্"),
        ("k,,k", "ক্ক"),
        ("kk,,", "ক্ক্"),
        (",,", "্"),
        (",,k", "্ক"),
        ("n,,d,,r,,", "ন্দ্র্"),
        // ড় is a consonant, so it takes a hasant
        ("kR,,", "কড়্"),
        // documented in data/rules/simplified_rules.md
        ("rr,,ka", "র্কা"),
        ("rrk,,Sh", "র্ক্ষ"),
    ] {
        assert_eq!(engine.transliterate(input), expected, "{input}");
    }
}

/// Structural guard: no output may carry a hasant that has no consonant before
/// it. This needs no expected output, so it keeps holding for inputs nobody
/// tabulated. A word-initial hasant is exempt — that is the documented
/// standalone `,,` marker.
#[test]
fn test_engine_never_emits_a_hasant_without_a_consonant_before_it() {
    let engine = ObadhEngine::new();
    let mut inputs: Vec<String> = Vec::new();

    for consonant in CONSONANTS {
        for vowel in VOWELS {
            for marker in MARKERS {
                inputs.push(format!("{consonant}{vowel}{marker},,"));
                inputs.push(format!("{consonant}{vowel}{marker},,k"));
            }
        }
    }

    assert!(inputs.len() > 3_000, "sweep too small: {}", inputs.len());

    let mut violations: Vec<String> = Vec::new();
    for input in &inputs {
        let output = engine.transliterate(input);
        let characters: Vec<char> = output.chars().collect();

        for (index, &character) in characters.iter().enumerate() {
            if character != HASANT {
                continue;
            }
            let Some(&previous) = index.checked_sub(1).and_then(|i| characters.get(i)) else {
                continue;
            };
            if !bears_hasant(previous) {
                violations.push(format!(
                    "{input:?} -> {output:?}: hasant on U+{:04X}",
                    previous as u32
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "{} of {} inputs put a hasant on a sign that cannot carry it, first 10:\n{}",
        violations.len(),
        inputs.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
