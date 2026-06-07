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
