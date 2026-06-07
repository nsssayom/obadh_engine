use clap::{Arg, ArgAction, Command};
use serde_json::{json, Value};
use std::env;
use std::io::{self, Read};
use std::time::{Duration, Instant};

use obadh_engine::engine::{Token, TokenType, Transliterator};

// Single source of version - using the crate version from Cargo.toml
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CLI with clap
    let matches = Command::new("obadh")
        .version(VERSION)
        .about("A deterministic Roman-to-Bengali transliteration engine")
        .arg(
            Arg::new("INPUT")
                .help("Input text to transliterate")
                .index(1),
        )
        .arg(
            Arg::new("debug")
                .short('d')
                .long("debug")
                .help("Output detailed information in JSON format")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Output more detailed information in JSON format")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("benchmark")
                .short('b')
                .long("benchmark")
                .help("Run benchmark with N iterations")
                .num_args(0..=1)
                .default_missing_value("1")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("pretty")
                .short('p')
                .long("pretty")
                .help("Pretty-print the JSON output (only used with --debug or --verbose)")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    // Get command line flags
    let debug_mode = matches.get_flag("debug");
    let verbose_mode = matches.get_flag("verbose");
    let pretty_print = matches.get_flag("pretty");
    let benchmark_iterations = matches.get_one::<usize>("benchmark").copied();

    // Get the input text from arguments or stdin
    let input = if let Some(text) = matches.get_one::<String>("INPUT") {
        text.clone()
    } else {
        // Try to read from stdin
        let mut buffer = String::new();
        let bytes_read = io::stdin().read_to_string(&mut buffer)?;

        if bytes_read == 0 {
            // No input provided, show usage
            let _ = Command::new("obadh")
                .version(VERSION)
                .about("A deterministic Roman-to-Bengali transliteration engine")
                .print_help();
            println!();
            return Ok(());
        }

        buffer
    };

    // Initialize the transliterator
    let transliterator = Transliterator::new();

    // Process based on the flags
    if let Some(iterations) = benchmark_iterations {
        // Benchmark mode
        benchmark(
            &transliterator,
            &input,
            iterations,
            debug_mode || verbose_mode,
            pretty_print,
        )
    } else if debug_mode || verbose_mode {
        // Debug/verbose mode with JSON output
        process_json_output(&transliterator, &input, verbose_mode, pretty_print)
    } else {
        // Default mode: Simple output with just the transliterated text
        let result = transliterator.transliterate(&input);
        println!("{}", result);
        Ok(())
    }
}

/// Process text with JSON output for debug/verbose mode
fn process_json_output(
    transliterator: &Transliterator,
    input: &str,
    verbose: bool,
    pretty_print: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let profile = profile_transliteration(transliterator, input);

    // Create JSON output
    let mut output_json = json!({
        "input": input,
        "output": profile.output,
        "performance": {
            "total_ms": format_duration(profile.total_duration()),
            "sanitize_ms": format_duration(profile.sanitize_duration),
            "tokenize_ms": format_duration(profile.tokenize_duration),
            "transliterate_ms": format_duration(profile.transliterate_duration),
        }
    });

    // Add token analysis for verbose mode
    if verbose {
        if let Value::Object(ref mut map) = output_json {
            // Convert tokens to JSON structure with detailed analysis
            let token_analysis = profile
                .tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    let mut token_json = json!({
                        "content": token.content,
                        "type": format!("{:?}", token.token_type),
                        "position": token.position,
                        "transliterated": transliterator
                            .transliterate_token_at(&profile.tokens, index)
                    });

                    // If it's a word, include phonetic analysis
                    if token.token_type == TokenType::Word {
                        let phonetic_units = transliterator.tokenize_phonetic(&token.content);
                        let units_json = phonetic_units
                            .iter()
                            .map(|unit| {
                                json!({
                                    "text": unit.text,
                                    "type": format!("{:?}", unit.unit_type),
                                    "position": unit.position
                                })
                            })
                            .collect::<Vec<_>>();

                        if let Value::Object(ref mut token_map) = token_json {
                            token_map.insert("phonetic_units".to_string(), json!(units_json));
                        }
                    }

                    token_json
                })
                .collect::<Vec<_>>();

            map.insert("token_analysis".to_string(), json!(token_analysis));
        }
    }

    // Output the result
    if pretty_print {
        println!("{}", serde_json::to_string_pretty(&output_json)?);
    } else {
        println!("{}", serde_json::to_string(&output_json)?);
    }

    Ok(())
}

/// Format Duration to milliseconds with decimal precision
fn format_duration(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Run benchmark with multiple iterations
fn benchmark(
    transliterator: &Transliterator,
    input: &str,
    iterations: usize,
    json_output: bool,
    pretty_print: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize timing variables
    let mut total_duration = Duration::new(0, 0);
    let mut sanitize_duration = Duration::new(0, 0);
    let mut tokenize_duration = Duration::new(0, 0);
    let mut transliterate_duration = Duration::new(0, 0);

    // Run the benchmark
    for _ in 0..iterations {
        let profile = profile_transliteration(transliterator, input);
        sanitize_duration += profile.sanitize_duration;
        tokenize_duration += profile.tokenize_duration;
        transliterate_duration += profile.transliterate_duration;
        total_duration += profile.total_duration();
    }

    // Calculate averages
    let avg_total = total_duration / iterations as u32;
    let avg_sanitize = sanitize_duration / iterations as u32;
    let avg_tokenize = tokenize_duration / iterations as u32;
    let avg_transliterate = transliterate_duration / iterations as u32;

    // Output benchmark results
    let transliterated = transliterator.transliterate(input);

    if json_output {
        // JSON output for benchmark results
        let benchmark_json = json!({
            "input": input,
            "output": transliterated,
            "benchmark": {
                "iterations": iterations,
                "avg_total_ms": format_duration(avg_total),
                "avg_sanitize_ms": format_duration(avg_sanitize),
                "avg_tokenize_ms": format_duration(avg_tokenize),
                "avg_transliterate_ms": format_duration(avg_transliterate),
                "total_run_time_ms": format_duration(total_duration),
            }
        });

        if pretty_print {
            println!("{}", serde_json::to_string_pretty(&benchmark_json)?);
        } else {
            println!("{}", serde_json::to_string(&benchmark_json)?);
        }
    } else {
        // Simple text output for benchmark results
        println!("Translation: {}", transliterated);
        println!("Benchmark results ({} iterations):", iterations);
        println!("  Average total time: {:.4} ms", format_duration(avg_total));
        println!(
            "  Average sanitize time: {:.4} ms",
            format_duration(avg_sanitize)
        );
        println!(
            "  Average tokenize time: {:.4} ms",
            format_duration(avg_tokenize)
        );
        println!(
            "  Average transliterate time: {:.4} ms",
            format_duration(avg_transliterate)
        );
        println!(
            "  Total run time: {:.4} ms",
            format_duration(total_duration)
        );
    }

    Ok(())
}

struct TransliterationProfile {
    output: String,
    tokens: Vec<Token>,
    sanitize_duration: Duration,
    tokenize_duration: Duration,
    transliterate_duration: Duration,
}

impl TransliterationProfile {
    fn total_duration(&self) -> Duration {
        self.sanitize_duration + self.tokenize_duration + self.transliterate_duration
    }
}

fn profile_transliteration(transliterator: &Transliterator, input: &str) -> TransliterationProfile {
    let sanitize_start = Instant::now();
    let sanitized = transliterator.sanitize(input);
    let sanitize_duration = sanitize_start.elapsed();

    match sanitized {
        Ok(sanitized) => {
            let tokenize_start = Instant::now();
            let tokens = transliterator.tokenize(&sanitized);
            let tokenize_duration = tokenize_start.elapsed();

            let transliterate_start = Instant::now();
            let output = transliterator.transliterate_tokens(&tokens);
            let transliterate_duration = transliterate_start.elapsed();

            TransliterationProfile {
                output,
                tokens,
                sanitize_duration,
                tokenize_duration,
                transliterate_duration,
            }
        }
        Err(_) => TransliterationProfile {
            output: input.to_string(),
            tokens: Vec::new(),
            sanitize_duration,
            tokenize_duration: Duration::ZERO,
            transliterate_duration: Duration::ZERO,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_total_is_the_sum_of_current_iteration_phases() {
        let profile = TransliterationProfile {
            output: String::new(),
            tokens: Vec::new(),
            sanitize_duration: Duration::from_micros(3),
            tokenize_duration: Duration::from_micros(5),
            transliterate_duration: Duration::from_micros(7),
        };

        assert_eq!(profile.total_duration(), Duration::from_micros(15));
    }

    #[test]
    fn profiled_output_matches_normal_transliteration() {
        let transliterator = Transliterator::new();
        let input = "strI bhakt prokash korchhi 12.34";
        let profile = profile_transliteration(&transliterator, input);

        assert_eq!(profile.output, transliterator.transliterate(input));
        assert!(!profile.tokens.is_empty());
    }
}
