use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use obadh_engine::{ObadhEngine, Tokenizer};
use std::time::Duration;

const RULE_STRESS_TEXT: &str =
    "kha gha kkA kko rrka rrko kShya k,,Ya n,,d,,rA songskrriti bidyuT`` rrT``sa 123.45";
const CONJUNCT_STRESS_WORD: &str = "rrkShkShmyntrngghya";
const MIXED_RULE_TEXT: &str =
    "kA khA gA. rrkSh rrT``sa; k,,y k,,w m,,w,,ra\nngga ngghAt jNG jn 123.45";
const LENIENT_MIXED_TEXT: &str = "ami😀 12.34 Taka. rZyab🔥 rrkSh 1.a2 songskrriti🚫";

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

criterion_group! {
    name = hot_path;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = bench_tokenizer, bench_transliterator
}
criterion_main!(hot_path);
