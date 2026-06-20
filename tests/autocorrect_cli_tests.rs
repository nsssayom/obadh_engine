use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[test]
fn autocorrect_cli_builds_inspects_and_evaluates_artifacts() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-roundtrip");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let bangla_eval = workspace.path("bangla-eval.tsv");
    let roman_eval = workspace.path("roman-eval.tsv");

    fs::write(&lexicon_tsv, "আমি\t100\nআমার\t80\nবিজ্ঞান\t70\nকিরণ\t60\n")
        .expect("lexicon fixture should write");
    fs::write(&bangla_eval, "আমী\tআমি\nবিজান\tবিজ্ঞান\n").expect("bangla eval fixture should write");
    fs::write(&roman_eval, "biggan\tবিজ্ঞান\nami\tআমি\n").expect("roman eval fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));
    let build_json = json_stdout(&build);
    assert_eq!(build_json["entries"], 4);
    assert_eq!(build_json["input_rows"], 4);
    assert_eq!(build_json["duplicate_rows"], 0);
    assert_eq!(build_json["total_frequency"], 310);
    assert_eq!(build_json["max_frequency"], 100);
    assert!(build_json["trie_nodes"].as_u64().unwrap() >= 4);
    assert_eq!(build_json["skeleton_keys"], 4);
    assert!(build_json["skeleton_delete_keys"].as_u64().unwrap() >= 4);
    assert!(artifact.exists());

    let inspect = run_obadh_autocorrect(["inspect-lexicon", "--input", path_str(&artifact)]);
    assert!(inspect.status.success(), "stderr: {}", stderr(&inspect));
    let inspect_json = json_stdout(&inspect);
    assert_eq!(inspect_json["entries"], 4);
    assert_eq!(inspect_json["artifact_bytes"], build_json["artifact_bytes"]);
    assert_eq!(inspect_json["skeleton_keys"], build_json["skeleton_keys"]);
    assert_eq!(
        inspect_json["skeleton_delete_keys"],
        build_json["skeleton_delete_keys"]
    );

    let bangla_report = run_obadh_autocorrect([
        "eval",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&bangla_eval),
        "--input-kind",
        "bangla",
    ]);
    assert!(
        bangla_report.status.success(),
        "stderr: {}",
        stderr(&bangla_report)
    );
    let bangla_json = json_stdout(&bangla_report);
    assert_eq!(bangla_json["total"], 2);
    assert_eq!(bangla_json["target_in_lexicon"], 2);
    assert_eq!(bangla_json["target_in_candidates"], 2);
    assert_eq!(bangla_json["target_lexicon_coverage"], 1.0);
    assert_eq!(bangla_json["candidate_recall_given_target_in_lexicon"], 1.0);
    assert_eq!(bangla_json["suggestion_recall"], 2);
    assert!(
        bangla_json["mean_reciprocal_rank"].as_f64().unwrap() > 0.0,
        "expected candidates to be ranked for typo pairs: {bangla_json}"
    );

    let roman_report = run_obadh_autocorrect([
        "eval",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&roman_eval),
        "--input-kind",
        "roman",
    ]);
    assert!(
        roman_report.status.success(),
        "stderr: {}",
        stderr(&roman_report)
    );
    let roman_json = json_stdout(&roman_report);
    assert_eq!(roman_json["total"], 2);
    assert_eq!(roman_json["target_in_lexicon"], 2);
    assert_eq!(roman_json["final_output_correct"], 2);
    assert_eq!(roman_json["baseline_accuracy"], 1.0);
}

#[test]
fn autocorrect_cli_builds_lexicon_from_multiple_sources() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-merge-lexicons");
    let primary = workspace.path("primary.tsv");
    let secondary = workspace.path("secondary.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let pairs = workspace.path("pairs.tsv");
    let output = workspace.path("candidates.jsonl");

    fs::write(&primary, "আমি\t100\nবিজ্ঞান\t70\n").expect("primary lexicon fixture should write");
    fs::write(&secondary, "আমি\t50\nকিরণ\t40\n").expect("secondary lexicon fixture should write");
    fs::write(&pairs, "আমী\tআমি\n").expect("pair fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&primary),
        "--input",
        path_str(&secondary),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));
    let build_json = json_stdout(&build);
    assert_eq!(build_json["inputs"].as_array().unwrap().len(), 2);
    assert_eq!(build_json["input_rows"], 4);
    assert_eq!(build_json["duplicate_rows"], 1);
    assert_eq!(build_json["entries"], 3);
    assert_eq!(build_json["total_frequency"], 260);
    assert_eq!(build_json["max_frequency"], 150);

    let export = run_obadh_autocorrect([
        "export-candidates",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&pairs),
        "--output",
        path_str(&output),
        "--input-kind",
        "bangla",
    ]);
    assert!(export.status.success(), "stderr: {}", stderr(&export));

    let jsonl = fs::read_to_string(&output).expect("candidate export should read");
    let record = serde_json::from_str::<Value>(jsonl.trim()).expect("JSONL line should parse");
    let merged = record["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "আমি")
        .expect("merged target candidate should be present");
    assert_eq!(merged["frequency"], 150);
}

#[test]
fn autocorrect_cli_rejects_non_bangla_lexicon_words_by_default() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-rejects-pollution");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    fs::write(&lexicon_tsv, "আমি\t100\nhello\t1\n").expect("lexicon fixture should write");

    let output = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);

    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("non-Bangla lexicon word"),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn autocorrect_cli_extracts_and_audits_clean_bangla_word_frequencies() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-audit");
    let corpus = workspace.path("corpus.txt");
    let extracted = workspace.path("extracted.tsv");
    fs::write(&corpus, "আমি আমি তুমি। hello ১২৩। র‌্যাব র‌্যাব ্ । কিরণ")
        .expect("corpus fixture should write");

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&corpus),
        "--output",
        path_str(&extracted),
        "--min-frequency",
        "2",
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["token_count"], 6);
    assert_eq!(extract_json["unique_words"], 4);
    assert_eq!(extract_json["emitted_entries"], 2);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("আমি\t2"));
    assert!(extracted_tsv.contains("র‌্যাব\t2"));
    assert!(!extracted_tsv.contains("১২৩"));
    assert!(!extracted_tsv.contains("hello"));

    let audit = run_obadh_autocorrect(["audit-lexicon", "--input", path_str(&extracted)]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);
    assert_eq!(audit_json["rows"], 2);
    assert_eq!(audit_json["accepted_rows"], 2);
    assert_eq!(audit_json["unique_words"], 2);
    assert_eq!(audit_json["non_bangla_rows"], 0);
}

#[test]
fn autocorrect_cli_merges_lexicon_sources_with_explicit_invalid_drops() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-merge-clean");
    let primary = workspace.path("primary.tsv");
    let secondary = workspace.path("secondary.tsv");
    let merged = workspace.path("merged.tsv");

    fs::write(&primary, "আমি\t100\nhello\t10\nবিজ্ঞান\tbad\n")
        .expect("primary lexicon fixture should write");
    fs::write(&secondary, "আমি\t50\nকিরণ\t40\n").expect("secondary lexicon fixture should write");

    let strict = run_obadh_autocorrect([
        "merge-lexicon",
        "--input",
        path_str(&primary),
        "--input",
        path_str(&secondary),
        "--output",
        path_str(&merged),
    ]);
    assert!(!strict.status.success());
    assert!(
        stderr(&strict).contains("non-Bangla lexicon word"),
        "stderr: {}",
        stderr(&strict)
    );

    let merge = run_obadh_autocorrect([
        "merge-lexicon",
        "--input",
        path_str(&primary),
        "--input",
        path_str(&secondary),
        "--output",
        path_str(&merged),
        "--drop-invalid",
    ]);
    assert!(merge.status.success(), "stderr: {}", stderr(&merge));
    let merge_json = json_stdout(&merge);
    assert_eq!(merge_json["rows"], 5);
    assert_eq!(merge_json["accepted_rows"], 3);
    assert_eq!(merge_json["duplicate_rows"], 1);
    assert_eq!(merge_json["dropped_rows"], 2);
    assert_eq!(merge_json["non_bangla_rows"], 1);
    assert_eq!(merge_json["invalid_frequency_rows"], 1);
    assert_eq!(merge_json["unique_words"], 2);
    assert_eq!(merge_json["emitted_entries"], 2);
    assert_eq!(merge_json["total_frequency"], 190);

    let merged_tsv = fs::read_to_string(&merged).expect("merged TSV should read");
    assert!(merged_tsv.contains("আমি\t150"));
    assert!(merged_tsv.contains("কিরণ\t40"));
    assert!(!merged_tsv.contains("hello"));
    assert!(!merged_tsv.contains("বিজ্ঞান"));
}

#[test]
fn autocorrect_cli_audits_bangla_pair_quality() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-audit-bangla-pairs");
    let pairs = workspace.path("bangla-pairs.tsv");
    fs::write(
        &pairs,
        "আমী\tআমি\nআমী\tআমি\nআমি\tআমি\nhello\tআমি\nআমী\thello\nআমী\tআমি\textra\n",
    )
    .expect("pair fixture should write");

    let audit = run_obadh_autocorrect([
        "audit-pairs",
        "--input",
        path_str(&pairs),
        "--input-kind",
        "bangla",
        "--pretty",
    ]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);

    assert_eq!(audit_json["rows"], 6);
    assert_eq!(audit_json["accepted_rows"], 3);
    assert_eq!(audit_json["unique_pairs"], 2);
    assert_eq!(audit_json["duplicate_rows"], 1);
    assert_eq!(audit_json["identity_rows"], 1);
    assert_eq!(audit_json["baseline_exact_rows"], 1);
    assert_eq!(audit_json["non_bangla_source_rows"], 1);
    assert_eq!(audit_json["non_bangla_expected_rows"], 1);
    assert_eq!(audit_json["malformed_rows"], 1);
    assert!(
        audit_json["baseline_mean_edit_cost"].as_f64().unwrap() > 0.0,
        "expected typo rows to produce a non-zero mean edit cost: {audit_json}"
    );
}

#[test]
fn autocorrect_cli_audits_clean_roman_pair_quality() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-audit-roman-pairs");
    let pairs = workspace.path("roman-pairs.tsv");
    fs::write(&pairs, "biggan\tবিজ্ঞান\nami\tআমি\nআমি\tআমি\nami!\tআমি\n")
        .expect("pair fixture should write");

    let audit = run_obadh_autocorrect([
        "audit-pairs",
        "--input",
        path_str(&pairs),
        "--input-kind",
        "roman",
    ]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);

    assert_eq!(audit_json["rows"], 4);
    assert_eq!(audit_json["accepted_rows"], 2);
    assert_eq!(audit_json["unique_pairs"], 2);
    assert_eq!(audit_json["baseline_exact_rows"], 2);
    assert_eq!(audit_json["non_roman_source_rows"], 2);
    assert_eq!(audit_json["non_bangla_expected_rows"], 0);
    assert_eq!(audit_json["baseline_exact_rate"], 1.0);
}

#[test]
fn autocorrect_cli_exports_labeled_candidate_jsonl() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-export-candidates");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let pairs = workspace.path("pairs.tsv");
    let output = workspace.path("candidates.jsonl");

    fs::write(&lexicon_tsv, "আমি\t1000\nবিজ্ঞান\t900\n").expect("lexicon fixture should write");
    fs::write(&pairs, "আমী\tআমি\nবিজ্ঞান\tবিজ্ঞান\n").expect("pair fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let export = run_obadh_autocorrect([
        "export-candidates",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&pairs),
        "--output",
        path_str(&output),
        "--input-kind",
        "bangla",
        "--max-candidates",
        "1",
    ]);
    assert!(export.status.success(), "stderr: {}", stderr(&export));
    let export_json = json_stdout(&export);
    assert_eq!(export_json["rows"], 2);
    assert_eq!(export_json["target_in_lexicon_rows"], 2);
    assert_eq!(export_json["target_present_rows"], 2);
    assert!(export_json["candidate_rows"].as_u64().unwrap() >= 2);

    let jsonl = fs::read_to_string(&output).expect("candidate export should read");
    let records = jsonl
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("JSONL line should parse"))
        .collect::<Vec<_>>();

    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["candidates"].as_array().unwrap().len(), 1);
    assert_eq!(records[0]["baseline"], "আমী");
    assert_eq!(records[0]["baseline_exact"], false);
    assert_eq!(records[0]["target_in_lexicon"], true);
    assert!(records[0]["target_rank"].as_u64().unwrap() >= 1);
    let labeled = records[0]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["label"] == true)
        .expect("expected target candidate should be labeled");
    assert_eq!(labeled["text"], "আমি");
    assert_eq!(labeled["source"], "lexicon_edit");
    assert_eq!(labeled["features"].as_array().unwrap().len(), 10);

    assert_eq!(records[1]["baseline"], "বিজ্ঞান");
    assert_eq!(records[1]["baseline_exact"], true);
}

struct TempWorkspace {
    root: PathBuf,
}

impl TempWorkspace {
    fn new(label: &str) -> Self {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{label}-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&root).expect("temp workspace should be created");
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn run_obadh_autocorrect<const N: usize>(args: [&str; N]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_obadh-autocorrect"))
        .args(args)
        .output()
        .expect("obadh-autocorrect should run")
}

fn json_stdout(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("stdout should be JSON: {error}; stdout={}", stdout(output)))
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("test path should be valid UTF-8")
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
