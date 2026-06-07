//! Definitions for Bengali special symbols and punctuation
//!
//! This file contains mappings for Bengali special symbols and punctuation.

use super::insert_unique;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A Roman input rule and its Bengali symbol output.
pub type SymbolRule = (&'static str, &'static str);

const SYMBOL_RULES: &[SymbolRule] = &[
    (".", "।"), // Bengali full stop (Dari)
    ("$", "৳"), // BDT symbol
];

fn build_symbols() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::with_capacity(SYMBOL_RULES.len());

    for &(roman, bengali) in SYMBOL_RULES {
        insert_unique(&mut map, "symbol", roman, bengali);
    }

    map
}

/// Returns the ordered static symbol rule table.
pub const fn symbol_rules() -> &'static [SymbolRule] {
    SYMBOL_RULES
}

/// Look up a Bengali symbol from a Roman rule key without constructing or hashing a map.
pub fn symbol_value(roman: &str) -> Option<&'static str> {
    match roman {
        "." => Some("।"),
        "$" => Some("৳"),
        _ => None,
    }
}

/// Returns a shared map of Bengali punctuation and special symbols.
pub fn symbols_static() -> &'static HashMap<&'static str, &'static str> {
    static INSTANCE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    INSTANCE.get_or_init(build_symbols)
}

/// Returns a map of Bengali punctuation and special symbols
pub fn symbols() -> HashMap<&'static str, &'static str> {
    symbols_static().clone()
}
