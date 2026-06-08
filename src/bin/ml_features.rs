use clap::{Arg, ArgAction, Command};
use obadh_engine::ObadhEngine;
use std::io::{self, BufRead};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = cli().get_matches();
    let pretty = matches.get_flag("pretty");
    let engine = ObadhEngine::new();

    if let Some(text) = matches.get_one::<String>("INPUT") {
        print_feature_doc(&engine, text, pretty)?;
        return Ok(());
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        print_feature_doc(&engine, &line, false)?;
    }

    Ok(())
}

fn cli() -> Command {
    Command::new("obadh-ml-features")
        .version(VERSION)
        .about("Emit versioned Obadh ML feature JSON")
        .arg(
            Arg::new("INPUT")
                .help("Input text to analyze. Reads newline-delimited stdin when omitted.")
                .index(1),
        )
        .arg(
            Arg::new("pretty")
                .short('p')
                .long("pretty")
                .help("Pretty-print single-input JSON output")
                .action(ArgAction::SetTrue),
        )
}

fn print_feature_doc(
    engine: &ObadhEngine,
    text: &str,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let features = engine.ml_features(text);
    if pretty {
        println!("{}", serde_json::to_string_pretty(&features)?);
    } else {
        println!("{}", serde_json::to_string(&features)?);
    }
    Ok(())
}
