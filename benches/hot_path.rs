use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use obadh_engine::{
    AutocorrectConfig, AutocorrectEngine, CharCandidateReranker, CorrectionRequest, Lexicon,
    LexiconEntry, ObadhEngine, Tokenizer,
};
use std::time::Duration;

const RULE_STRESS_TEXT: &str =
    "kha gha kkA kko rrka rrko kShya k,,Ya n,,d,,rA songskrriti bidyuT`` rrT``sa 123.45";
const CONJUNCT_STRESS_WORD: &str = "rrkShkShmyntrngghya";
const MIXED_RULE_TEXT: &str =
    "kA khA gA. rrkSh rrT``sa; k,,y k,,w m,,w,,ra\nngga ngghAt jNG jn 123.45";
const LENIENT_MIXED_TEXT: &str = "ami😀 12.34 Taka. rZyab🔥 rrkSh 1.a2 songskrriti🚫";
const AUTOCORRECT_INPUT: &str = "কীরন";
const SHIPPED_AUTOCORRECT_LEXICON: &[u8] = include_bytes!("../www/assets/autocorrect/bn.lex");
const SHIPPED_CHAR_RERANKER: &[u8] =
    include_bytes!("../www/assets/autocorrect/char_candidate_reranker.json");

fn bench_tokenizer(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();

    let mut group = c.benchmark_group("tokenizer");
    group.throughput(Throughput::Bytes(RULE_STRESS_TEXT.len() as u64));
    group.bench_function("tokenize_text_rule_stress", |b| {
        b.iter(|| tokenizer.tokenize_text(black_box(RULE_STRESS_TEXT)));
    });

    group.throughput(Throughput::Bytes(CONJUNCT_STRESS_WORD.len() as u64));
    group.bench_function("tokenize_word_conjunct_stress", |b| {
        b.iter(|| tokenizer.tokenize_word(black_box(CONJUNCT_STRESS_WORD)));
    });
    group.finish();
}

fn bench_transliterator(c: &mut Criterion) {
    let engine = ObadhEngine::new();
    let tokens = engine.tokenize(MIXED_RULE_TEXT);

    let mut group = c.benchmark_group("transliterator");
    group.throughput(Throughput::Bytes(MIXED_RULE_TEXT.len() as u64));
    group.bench_function("transliterate_mixed_rule_text", |b| {
        b.iter(|| engine.transliterate(black_box(MIXED_RULE_TEXT)));
    });

    group.bench_function("render_pre_tokenized_mixed_rule_text", |b| {
        b.iter(|| engine.transliterate_tokens(black_box(&tokens)));
    });

    group.throughput(Throughput::Bytes(LENIENT_MIXED_TEXT.len() as u64));
    group.bench_function("transliterate_lenient_mixed_invalid_text", |b| {
        b.iter(|| engine.transliterate_lenient(black_box(LENIENT_MIXED_TEXT)));
    });
    group.finish();
}

fn bench_autocorrect(c: &mut Criterion) {
    let obadh = ObadhEngine::new();
    let autocorrect = AutocorrectEngine::from_entries(stress_lexicon_entries());
    let request = CorrectionRequest::new(AUTOCORRECT_INPUT);
    let shipped_lexicon = Lexicon::from_compact_bytes(SHIPPED_AUTOCORRECT_LEXICON)
        .expect("shipped autocorrect lexicon should load");
    let shipped_autocorrect = AutocorrectEngine::with_config(
        shipped_lexicon,
        AutocorrectConfig {
            max_candidates: 512,
            search_known_input: true,
            max_skeleton_candidates: 512,
            ..AutocorrectConfig::default()
        },
    );
    let shipped_reranker = CharCandidateReranker::from_json_bytes(SHIPPED_CHAR_RERANKER)
        .expect("shipped char reranker should load");

    let mut group = c.benchmark_group("autocorrect");
    group.throughput(Throughput::Elements(1));
    group.bench_function("decide_stress_lexicon_input", |b| {
        b.iter(|| autocorrect.decide(black_box(request.clone())));
    });

    group.bench_function("obadh_request_then_decide_stress_lexicon", |b| {
        b.iter(|| {
            let request = obadh.autocorrect_request(black_box("biggan"));
            autocorrect.decide(black_box(request))
        });
    });

    group.bench_function("shipped_roman_request_decide_and_rerank_512", |b| {
        b.iter(|| {
            let request = obadh.autocorrect_request(black_box("okalpokko"));
            let decision = shipped_autocorrect.decide(black_box(request));
            shipped_reranker
                .rank_candidates(black_box("okalpokko"), black_box(&decision.candidates))
        });
    });
    group.finish();
}

fn stress_lexicon_entries() -> Vec<LexiconEntry> {
    let heads = [
        "ক", "খ", "গ", "ঘ", "চ", "জ", "ট", "ড", "ত", "দ", "ন", "প", "ব", "ম", "য", "র",
    ];
    let tails = [
        "ান", "িন", "ীন", "ুন", "েন", "োন", "ার", "ির", "ের", "াল", "িল", "ুল", "েল", "াম", "িম", "ুম",
    ];

    heads
        .iter()
        .flat_map(|head| tails.iter().map(move |tail| format!("{head}{tail}")))
        .enumerate()
        .map(|(index, word)| LexiconEntry::new(word, 10_000_u32.saturating_sub(index as u32)))
        .collect()
}

criterion_group! {
    name = hot_path;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = bench_tokenizer, bench_transliterator, bench_autocorrect
}
criterion_main!(hot_path);
