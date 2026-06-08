use obadh_engine::{ObadhEngine, FEATURE_SCHEMA_VERSION, FEATURE_SLOTS_PER_UNIT};
use std::process::{Command, Stdio};

#[test]
fn ml_features_are_versioned_and_expand_word_units_for_ctc() {
    let engine = ObadhEngine::new();
    let features = engine.ml_features("aYp biggan");

    assert_eq!(features.schema, FEATURE_SCHEMA_VERSION);
    assert_eq!(features.deterministic, "অ্যাপ বিজ্ঞান");
    assert!(features.accepted);
    assert!(features.sanitization_error.is_none());

    let word_tokens = features
        .tokens
        .iter()
        .filter(|token| token.token_type == "word")
        .collect::<Vec<_>>();

    assert_eq!(word_tokens.len(), 2);
    for token in word_tokens {
        assert_eq!(
            token.slots.len(),
            token.units.len() * FEATURE_SLOTS_PER_UNIT
        );
        assert!(token
            .slots
            .iter()
            .any(|slot| slot.slot_type == "main" && slot.feature_key.contains('|')));
    }
}

#[test]
fn ml_features_report_rejected_input_without_cleaning_it() {
    let engine = ObadhEngine::new();
    let features = engine.ml_features("aYp🙂");

    assert_eq!(features.schema, FEATURE_SCHEMA_VERSION);
    assert!(!features.accepted);
    assert_eq!(features.deterministic, "aYp🙂");
    assert!(features.sanitization_error.is_some());
    assert!(features.tokens.is_empty());
}

#[test]
fn ml_feature_binary_streams_jsonl() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_obadh-ml-features"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("feature binary should spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin should be piped");
        writeln!(stdin, "aYp").expect("write first input");
        writeln!(stdin, "biggan").expect("write second input");
    }

    let output = child
        .wait_with_output()
        .expect("feature binary should exit");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    let lines = stdout.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("\"schema\":\"obadh.ml.features.v0\""));
    assert!(lines[0].contains("\"deterministic\":\"অ্যাপ\""));
    assert!(lines[1].contains("\"deterministic\":\"বিজ্ঞান\""));
}
