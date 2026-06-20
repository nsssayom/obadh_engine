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
    assert!(!correction.features.generated_roman_candidate);
    assert_eq!(correction.features.as_i16_array()[0], 1);
}

#[test]
fn autocorrect_request_generates_roman_missing_vowel_neighbors() {
    let obadh = ObadhEngine::new();
    let request = obadh.autocorrect_request("okalpokko");

    assert_eq!(request.current, "অকল্পক্ক");
    assert!(request
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "অকালপক্ক"));
    assert!(!request
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "অকল্পকক"));
}

#[test]
fn autocorrect_request_generates_two_gap_roman_missing_vowel_neighbors() {
    let obadh = ObadhEngine::new();
    let request = obadh.autocorrect_request("kmn");

    assert_eq!(request.current, "ক্মন");
    assert!(request
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "কেমন"));
    assert!(
        request.generated_candidates.len() <= 24,
        "generated candidates should stay bounded: {:?}",
        request.generated_candidates
    );
}

#[test]
fn autocorrect_request_generates_prioritized_sparse_roman_vowel_variants() {
    let obadh = ObadhEngine::new();
    let tomar = obadh.autocorrect_request("tmr");
    let tomake = obadh.autocorrect_request("tmk");
    let kothay = obadh.autocorrect_request("kthay");
    let jemon = obadh.autocorrect_request("jmn");
    let jabo = obadh.autocorrect_request("jbo");
    let korbo = obadh.autocorrect_request("krbo");

    assert!(tomar
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "তোমার"));
    assert!(tomake
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "তোমাকে"));
    assert!(kothay
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "কোথায়"));
    assert!(jemon
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "যেমন"));
    assert!(jabo
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "যাবো"));
    assert!(korbo
        .generated_candidates
        .iter()
        .any(|candidate| candidate == "করবো"));
    assert!(tomar.generated_candidates.len() <= 24);
    assert!(tomake.generated_candidates.len() <= 24);
    assert!(kothay.generated_candidates.len() <= 24);
    assert!(jemon.generated_candidates.len() <= 24);
    assert!(jabo.generated_candidates.len() <= 24);
    assert!(korbo.generated_candidates.len() <= 24);
}

#[test]
fn autocorrect_can_suggest_generated_roman_neighbor_without_lexicon_entry() {
    let obadh = ObadhEngine::new();
    let autocorrect = AutocorrectEngine::with_config(
        Lexicon::default(),
        AutocorrectConfig {
            max_candidates: 16,
            ..AutocorrectConfig::default()
        },
    );

    let decision = autocorrect.decide(obadh.autocorrect_request("okalpokko"));

    let candidate = decision
        .candidates
        .iter()
        .find(|candidate| candidate.text == "অকালপক্ক")
        .expect("deterministic Roman neighbor should be suggested");
    assert_eq!(candidate.source, CorrectionSource::RomanEdit);
    assert_eq!(candidate.features.source_id, 3);
    assert!(candidate.features.candidate_known);
    assert!(candidate.features.generated_roman_candidate);
}

#[test]
fn autocorrect_deduplicates_generated_candidates_against_lexicon_evidence() {
    let autocorrect = AutocorrectEngine::with_config(
        Lexicon::new([LexiconEntry::new("অকালপক্ক", 1)]),
        AutocorrectConfig {
            max_candidates: 16,
            max_skeleton_candidates: 16,
            ..AutocorrectConfig::default()
        },
    );

    let decision =
        autocorrect.decide(CorrectionRequest::new("অকল্পক্ক").with_generated_candidates(["অকালপক্ক"]));
    let candidates = decision
        .candidates
        .iter()
        .filter(|candidate| candidate.text == "অকালপক্ক")
        .collect::<Vec<_>>();

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].source, CorrectionSource::PhoneticSkeleton);
    assert!(candidates[0].features.generated_roman_candidate);
}
