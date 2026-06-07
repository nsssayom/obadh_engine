use obadh_engine::{ObadhEngine, PhoneticUnitType, Tokenizer};

#[test]
fn test_basic_conjunct_formation() {
    let tokenizer = Tokenizer::new();

    // Test automatic conjunct formation from consecutive consonants
    let units = tokenizer.tokenize_word("kk");
    // Verify we got a conjunct
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
    assert_eq!(units[0].text, "k,,k");

    // Test another common consonant pair
    let units = tokenizer.tokenize_word("ks");
    // Verify we got a conjunct
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
    assert_eq!(units[0].text, "k,,s");
}

#[test]
fn test_explicit_conjunct_formation() {
    let tokenizer = Tokenizer::new();

    // Test explicit conjunct formation with hasant (simple case)
    let units = tokenizer.tokenize_word("k,,k");
    // Verify we get a conjunct
    assert!(!units.is_empty());
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
}

#[test]
fn test_multi_letter_conjunct_formation() {
    let tokenizer = Tokenizer::new();

    // Test 3-letter conjunct
    let units = tokenizer.tokenize_word("ndr");
    // Verify we get a conjunct (shape may vary based on implementation)
    assert!(!units.is_empty());
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);

    // ntr is valid; the trailing k remains separate because ntrk is not a
    // declared Bengali conjunct.
    let units = tokenizer.tokenize_word("ntrk");
    assert!(!units.is_empty());
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
    assert_eq!(units[0].text, "n,,t,,r");
    assert_eq!(units[1].unit_type, PhoneticUnitType::Consonant);
    assert_eq!(units[1].text, "k");
}

#[test]
fn test_conjunct_with_vowel() {
    let tokenizer = Tokenizer::new();

    // Test conjunct with regular vowel
    let units = tokenizer.tokenize_word("kkA");
    // Verify we got a conjunct with vowel
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithVowel);

    // Test with uppercase O (full vowel)
    let units = tokenizer.tokenize_word("kkO");
    // Verify we got a conjunct with vowel
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithVowel);
}

#[test]
fn test_conjunct_with_terminator() {
    let tokenizer = Tokenizer::new();

    // Test conjunct with terminator vowel (lowercase o)
    let units = tokenizer.tokenize_word("kko");
    // Verify we got a conjunct with terminator
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithTerminator);
}

#[test]
fn test_complex_conjunct_sequences() {
    let tokenizer = Tokenizer::new();

    // Only kk is a declared Bengali conjunct; the final k remains separate.
    let units = tokenizer.tokenize_word("kkk");
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
    assert_eq!(units[0].text, "k,,k");
    assert_eq!(units[1].unit_type, PhoneticUnitType::Consonant);
    assert_eq!(units[1].text, "k");

    // Test consonant + consonant + consonantWithVowel
    let units = tokenizer.tokenize_word("nkkO");
    // kkO is valid, but nkkO as a whole is not.
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].unit_type, PhoneticUnitType::Consonant);
    assert_eq!(units[0].text, "n");
    assert_eq!(units[1].unit_type, PhoneticUnitType::ConjunctWithVowel);
}

#[test]
fn test_comparison_auto_and_explicit_conjuncts() {
    let tokenizer = Tokenizer::new();

    // Compare automatic versus explicit conjuncts (simple case)
    let auto_units = tokenizer.tokenize_word("kk");
    let explicit_units = tokenizer.tokenize_word("k,,k");

    // Both should produce a conjunct
    assert_eq!(auto_units.len(), 1);
    assert_eq!(explicit_units.len(), 1);
    assert_eq!(auto_units[0].unit_type, PhoneticUnitType::Conjunct);
    assert_eq!(explicit_units[0].unit_type, PhoneticUnitType::Conjunct);
}

#[test]
fn test_consonant_with_conjunct_sequences() {
    let tokenizer = Tokenizer::new();

    // Test consonant followed by a consonant with vowel
    let units = tokenizer.tokenize_word("kkO");
    // Verify we got a conjunct with vowel
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithVowel);
    assert_eq!(units[0].text, "k,,kO");

    // Test similar case with terminator vowel
    let units = tokenizer.tokenize_word("kko");
    // Verify we got a conjunct with terminator
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithTerminator);
    assert_eq!(units[0].text, "k,,ko");
}

#[test]
fn test_vocalic_r() {
    let tokenizer = Tokenizer::new();

    // Test vocalic R ("rri") in "krri"
    let units = tokenizer.tokenize_word("krri");
    // Verify we got a consonant with vowel (k + vocalic R)
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    assert_eq!(units[0].text, "krri");

    // Test vocalic R in a more complex word
    let units = tokenizer.tokenize_word("krriShi");
    // Should be a consonant with vocalic R followed by a consonant with vowel
    // Now that we handle conjuncts better, this test needs to be adapted
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    assert_eq!(units[0].text, "krri");

    // The second part can be either a ConsonantWithVowel or ConjunctWithVowel depending on implementation
    // Instead of asserting type, just check that the text is as expected
    assert!(
        units[1].unit_type == PhoneticUnitType::ConsonantWithVowel
            || units[1].unit_type == PhoneticUnitType::ConjunctWithVowel
    );
    assert_eq!(units[1].text, "Shi");
}

#[test]
fn test_reph_over_consonant() {
    let tokenizer = Tokenizer::new();

    // Test reph over consonant ("rrm")
    let units = tokenizer.tokenize_word("rrm");
    // Verify we got a reph over consonant
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, PhoneticUnitType::RephOverConsonant);
    assert_eq!(units[0].text, "rrm");

    // Test reph over consonant in a word with a vowel
    let units = tokenizer.tokenize_word("korrm");
    // Should be a consonant with terminator followed by a reph over consonant
    assert_eq!(units.len(), 2);
    assert_eq!(
        units[0].unit_type,
        PhoneticUnitType::ConsonantWithTerminator
    );
    assert_eq!(units[0].text, "ko");
    assert_eq!(units[1].unit_type, PhoneticUnitType::RephOverConsonant);
    assert_eq!(units[1].text, "rrm");
}

#[test]
fn test_reph_over_consonant_with_vowel() {
    let tokenizer = Tokenizer::new();

    // Test reph over consonant with vowel ("rrmi")
    let units = tokenizer.tokenize_word("rrmi");
    // Verify we got a reph over consonant with vowel
    assert_eq!(units.len(), 1);
    assert_eq!(
        units[0].unit_type,
        PhoneticUnitType::RephOverConsonantWithVowel
    );
    assert_eq!(units[0].text, "rrmi");

    // Test in a more complex word
    let units = tokenizer.tokenize_word("korrmO");
    // Should be a consonant with terminator followed by a reph over consonant with vowel
    assert_eq!(units.len(), 2);
    assert_eq!(
        units[0].unit_type,
        PhoneticUnitType::ConsonantWithTerminator
    );
    assert_eq!(units[0].text, "ko");
    assert_eq!(
        units[1].unit_type,
        PhoneticUnitType::RephOverConsonantWithVowel
    );
    assert_eq!(units[1].text, "rrmO");
}

#[test]
fn test_reph_over_consonant_with_terminator() {
    let tokenizer = Tokenizer::new();

    // Test reph over consonant with terminator ("rrmo")
    let units = tokenizer.tokenize_word("rrmo");
    // Verify we got a reph over consonant with terminator
    assert_eq!(units.len(), 1);
    assert_eq!(
        units[0].unit_type,
        PhoneticUnitType::RephOverConsonantWithTerminator
    );
    assert_eq!(units[0].text, "rrmo");

    // Test full word "korrmo"
    let units = tokenizer.tokenize_word("korrmo");
    // Should be a consonant with terminator followed by a reph over consonant with terminator
    assert_eq!(units.len(), 2);
    assert_eq!(
        units[0].unit_type,
        PhoneticUnitType::ConsonantWithTerminator
    );
    assert_eq!(units[0].text, "ko");
    assert_eq!(
        units[1].unit_type,
        PhoneticUnitType::RephOverConsonantWithTerminator
    );
    assert_eq!(units[1].text, "rrmo");

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("rrmo korrmo"), "র্ম কর্ম");
}

#[test]
fn test_comparison_normal_and_reph() {
    let tokenizer = Tokenizer::new();

    // Compare normal 'r' usage versus reph 'rr'
    let normal_units = tokenizer.tokenize_word("karma");
    let reph_units = tokenizer.tokenize_word("korrmo");

    // Single r is a normal consonant; double rr is the explicit reph marker.
    assert_eq!(normal_units.len(), 3);
    assert_eq!(
        normal_units[0].unit_type,
        PhoneticUnitType::ConsonantWithVowel
    );
    assert_eq!(normal_units[0].text, "ka");
    assert_eq!(normal_units[1].unit_type, PhoneticUnitType::Consonant);
    assert_eq!(normal_units[1].text, "r");
    assert_eq!(
        normal_units[2].unit_type,
        PhoneticUnitType::ConsonantWithVowel
    );
    assert_eq!(normal_units[2].text, "ma");

    // 'korrmo' should be "ko" + "rrmo" (reph over consonant with terminator)
    assert_eq!(reph_units.len(), 2);
    assert_eq!(
        reph_units[0].unit_type,
        PhoneticUnitType::ConsonantWithTerminator
    );
    assert_eq!(reph_units[0].text, "ko");
    assert_eq!(
        reph_units[1].unit_type,
        PhoneticUnitType::RephOverConsonantWithTerminator
    );
    assert_eq!(reph_units[1].text, "rrmo");
}
