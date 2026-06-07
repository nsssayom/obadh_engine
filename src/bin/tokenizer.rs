use obadh_engine::Tokenizer;
use std::env;
use std::io::{self, BufRead};

fn main() {
    let tokenizer = Tokenizer::new();
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Process command line arguments
        let word = &args[1];
        process_word(&tokenizer, word);
    } else {
        // Process from stdin
        println!("Enter words to tokenize (one per line, Ctrl+D to exit):");
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(word) => {
                    if !word.trim().is_empty() {
                        process_word(&tokenizer, &word);
                    }
                }
                Err(e) => {
                    eprintln!("Error reading line: {}", e);
                    break;
                }
            }
        }
    }
}

fn process_word(tokenizer: &Tokenizer, word: &str) {
    let units = tokenizer.tokenize_word(word);

    println!("\nTokenization of '{}':", word);
    for unit in &units {
        println!("  Unit '{}' type: {:?}", unit.text, unit.unit_type);
    }

    // Show the total count
    println!("Total units: {}", units.len());
}
