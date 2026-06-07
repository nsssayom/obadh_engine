use obadh_engine::{ObadhEngine, Tokenizer};

#[test]
fn test_real_world_phola_examples() {
    // Create engine for transliteration testing
    let engine = ObadhEngine::new();
    let tokenizer = Tokenizer::new();

    // Test case 1: sohy => সহ্য
    let units = tokenizer.tokenize_word("sohy");
    // Verify structure
    assert!(!units.is_empty());

    // Test the transliteration result
    let result = engine.transliterate("sohy");
    assert_eq!(result, "সহ্য");

    // Test case 2: biSw => বিশ্ব
    let units = tokenizer.tokenize_word("biSw");
    // Verify structure
    assert!(!units.is_empty());

    // Test the transliteration result
    let result = engine.transliterate("biSw");
    assert_eq!(result, "বিশ্ব");
}

#[test]
fn test_specific_jo_phola_cases() {
    let engine = ObadhEngine::new();

    // Test various jo-phola (য-ফলা) cases
    let examples = [
        ("sohy", "সহ্য"),    // consonant + terminator + jo-phola
        ("sohyo", "সহ্যো"),  // consonant + terminator + jo-phola + vowel
        ("kohya", "কহ্যা"),   // with different consonant
        ("bhujy", "ভুজ্য"),   // consonant + vowel + jo-phola
        ("kriy", "ক্রিয়"),   // conjunct + vowel + jo-phola
        ("odhyoy", "অধ্যয়"), // complex form with multiple jo-pholas
    ];

    for (input, expected) in examples {
        let result = engine.transliterate(input);
        assert_eq!(result, expected);
    }
}

#[test]
fn test_specific_bo_phola_cases() {
    let engine = ObadhEngine::new();

    // Test various bo-phola (ব-ফলা) cases
    let examples = [
        ("biSw", "বিশ্ব"),    // consonant + vowel + bo-phola
        ("biSwas", "বিশ্বাস"), // consonant + vowel + bo-phola + vowel + consonant
        ("tw", "ত্ব"),        // simple bo-phola
        ("twa", "ত্বা"),       // bo-phola with vowel
        ("SwaSw", "শ্বাশ্ব"),   // multiple bo-pholas
    ];

    for (input, expected) in examples {
        let result = engine.transliterate(input);
        assert_eq!(result, expected);
    }
}

#[test]
fn test_both_phola_in_one_word() {
    let engine = ObadhEngine::new();

    // Test words that have both jo-phola and bo-phola
    let examples = [
        ("Swy", "শ্বয়"), // sequential pholas
    ];

    for (input, expected) in examples {
        let result = engine.transliterate(input);
        assert_eq!(result, expected);
    }
}

#[test]
fn test_composable_phola_orthography() {
    let engine = ObadhEngine::new();

    let examples = [("SwayattaSw", "শ্বায়ত্তশ্ব"), ("dwitiyw", "দ্বিতীয়")];

    for (input, expected) in examples {
        let result = engine.transliterate(input);
        assert_eq!(result, expected);
    }
}

#[test]
fn test_specific_phola_examples() {
    let engine = ObadhEngine::new();

    // Test case 1: sohy => সহ্য
    let result = engine.transliterate("sohy");
    assert_eq!(result, "সহ্য");

    // Test case 2: biSw => বিশ্ব
    let result = engine.transliterate("biSw");
    assert_eq!(result, "বিশ্ব");
}

#[test]
fn test_vocalic_r_case() {
    let engine = ObadhEngine::new();

    // Test krri => কৃ
    let result = engine.transliterate("krri");
    assert_eq!(result, "কৃ");
}
