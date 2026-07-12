use obadh_engine::{
    AutocorrectConfig, AutocorrectEngine, CorrectionRequest, CorrectionSource, Lexicon,
    LexiconEntry, ObadhEngine,
};

#[test]
fn autocorrect_preserves_exact_obadh_output() {
    let obadh = ObadhEngine::new();
    let autocorrect = AutocorrectEngine::from_entries([LexiconEntry::new("বিজ্ঞান", 10_000)]);
    let request = obadh.autocorrect_request("biggan");

    let decision = autocorrect.decide(request.clone());

    assert_eq!(request.current, "বিজ্ঞান");
    assert_eq!(request.roman_input.as_deref(), Some("biggan"));
    assert_eq!(request.obadh_output.as_deref(), Some("বিজ্ঞান"));
    assert_eq!(decision.replacement, None);
    assert_eq!(decision.candidates[0].source, CorrectionSource::NoChange);
}

#[test]
fn autocorrect_can_fix_simple_bangla_vowel_typos() {
    let autocorrect = AutocorrectEngine::from_entries([
        LexiconEntry::new("আমি", 50_000),
        LexiconEntry::new("আম", 500),
    ]);

    let decision = autocorrect.decide(CorrectionRequest::new("আমী"));

    assert_eq!(
        decision
            .replacement
            .as_ref()
            .map(|candidate| candidate.text.as_str()),
        Some("আমি")
    );
}

#[test]
fn autocorrect_uses_suggestions_when_margin_is_low() {
    let autocorrect = AutocorrectEngine::with_config(
        Lexicon::new([LexiconEntry::new("আমি", 1)]),
        AutocorrectConfig {
            autocorrect_margin: 1_000,
            ..AutocorrectConfig::default()
        },
    );

    let decision = autocorrect.decide(CorrectionRequest::new("আমী"));

    assert!(decision
        .candidates
        .iter()
        .any(|candidate| candidate.text == "আমি"));
    assert_eq!(decision.replacement, None);
}

#[test]
fn autocorrect_candidate_list_is_bounded() {
    let autocorrect = AutocorrectEngine::with_config(
        Lexicon::new([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আম", 9),
            LexiconEntry::new("আমার", 8),
            LexiconEntry::new("আমরা", 7),
        ]),
        AutocorrectConfig {
            max_candidates: 2,
            max_edit_cost: 8,
            max_skeleton_candidates: 0,
            ..AutocorrectConfig::default()
        },
    );

    let decision = autocorrect.decide(CorrectionRequest::new("আমী"));

    assert_eq!(decision.candidates.len(), 2);
    assert_eq!(decision.candidates[0].text, "আমি");
    assert!(decision
        .candidates
        .iter()
        .all(|candidate| candidate.source == CorrectionSource::LexiconEdit));
}

#[test]
fn autocorrect_features_mark_obadh_baseline_candidate() {
    let autocorrect = AutocorrectEngine::from_entries([LexiconEntry::new("আমি", 50_000)]);

    let decision = autocorrect.decide(CorrectionRequest::new("আমী").with_obadh_output("আমী"));
    let keep = decision
        .candidates
        .iter()
        .find(|candidate| candidate.source == CorrectionSource::NoChange)
        .expect("no-change candidate should be present");
    let correction = decision
        .candidates
        .iter()
        .find(|candidate| candidate.text == "আমি")
        .expect("lexicon correction should be present");

    assert!(keep.features.obadh_baseline);
    assert!(!correction.features.obadh_baseline);
    assert!(!correction.features.input_known);
    assert!(correction.features.candidate_known);
    assert_eq!(correction.features.as_i16_array()[0], 1);
}

#[test]
fn autocorrect_suggests_prefix_completions_from_lexicon_frequency() {
    let autocorrect = AutocorrectEngine::with_config(
        Lexicon::new([
            LexiconEntry::new("কেমন", 225),
            LexiconEntry::new("কেমনি", 11),
            LexiconEntry::new("কেমনে", 10),
            LexiconEntry::new("যেমন", 247),
        ]),
        AutocorrectConfig {
            max_candidates: 8,
            max_prefix_candidates: 4,
            max_skeleton_candidates: 0,
            max_edit_cost: 0,
            ..AutocorrectConfig::default()
        },
    );

    let decision = autocorrect.decide(CorrectionRequest::new("কেম"));
    let completions = decision
        .candidates
        .iter()
        .filter(|candidate| candidate.source == CorrectionSource::PrefixCompletion)
        .map(|candidate| candidate.text.as_str())
        .collect::<Vec<_>>();

    assert_eq!(completions, vec!["কেমন", "কেমনি", "কেমনে"]);
    assert_eq!(decision.replacement, None);
    let kemon = decision
        .candidates
        .iter()
        .find(|candidate| candidate.text == "কেমন")
        .expect("prefix completion should include কেমন");
    assert_eq!(kemon.features.source_id, 4);
    assert_eq!(kemon.edit_cost.0, 0);
}

#[test]
fn loanword_artifact_fingerprint_is_stable_and_verifiable() {
    use obadh_engine::{
        artifact_fingerprint, build_loanword_bytes, verify_artifact_fingerprint, LoanwordEntry,
        LoanwordLexicon,
    };

    let bytes = build_loanword_bytes([
        LoanwordEntry {
            english: "license".to_string(),
            bangla: "লাইসেন্স".to_string(),
            frequency: 100,
        },
        LoanwordEntry {
            english: "server".to_string(),
            bangla: "সার্ভার".to_string(),
            frequency: 90,
        },
    ])
    .expect("loanword bytes should build");

    // The loaded lexicon reports the same fingerprint as the raw artifact bytes.
    let expected = artifact_fingerprint(&bytes);
    let lexicon = LoanwordLexicon::from_bytes(bytes.clone()).expect("lexicon should load");
    assert_eq!(lexicon.artifact_fingerprint(), expected);
    assert_ne!(expected, 0);

    // Verify passes on the true fingerprint and fails loudly on a wrong one,
    // reporting both sides — the stale-artifact signal a downstream gates on.
    assert!(verify_artifact_fingerprint(&bytes, expected).is_ok());
    let error = verify_artifact_fingerprint(&bytes, expected ^ 1).unwrap_err();
    assert_eq!(error.actual, expected);
    assert_eq!(error.expected, expected ^ 1);

    // A different loanword set fingerprints differently.
    let other = build_loanword_bytes([LoanwordEntry {
        english: "license".to_string(),
        bangla: "লাইসেন্স".to_string(),
        frequency: 100,
    }])
    .expect("loanword bytes should build");
    assert_ne!(artifact_fingerprint(&other), expected);
}
