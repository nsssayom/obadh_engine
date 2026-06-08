//! Tokenizer for the Obadh Engine
//!
//! This module provides functionality to tokenize input text into words
//! and letters/phonemes for processing by the transliteration engine.

mod conjunct_runs;
mod diacritics;
mod explicit_hasant;
mod forms;
mod long_iya;
mod normalization;
mod parts;
mod patterns;
mod scan_hints;
mod scanner;

use diacritics::{append_trailing_diacritics, split_trailing_diacritics};
use forms::identify_complex_forms;
use patterns::{phonetic_pattern_trie_static, PatternTrie};
use scan_hints::WordScanHints;

/// Types of tokens that can be identified
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TokenType {
    /// A standard word token
    Word,
    /// A punctuation mark
    Punctuation,
    /// A whitespace token
    Whitespace,
    /// A numeric token
    Number,
    /// A special symbol
    Symbol,
}

/// A token from the input text
#[derive(Debug, Clone)]
pub struct Token {
    /// The content of the token
    pub content: String,
    /// The type of the token
    pub token_type: TokenType,
    /// The position of the token in the original text
    pub position: usize,
}

/// Represents a sequence of phonetic components that make up a word
#[derive(Debug, Clone)]
pub struct PhoneticUnit {
    /// The original text
    pub text: String,
    /// What type of phonetic unit this is
    pub unit_type: PhoneticUnitType,
    /// Position in the original word
    pub position: usize,
}

/// Types of phonetic units in Bengali transliteration
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PhoneticUnitType {
    /// Single consonant
    Consonant,
    /// Vowel
    Vowel,
    /// A terminating vowel like 'o' that completes syllables
    TerminatingVowel,
    /// A consonant with a vowel modifier
    ConsonantWithVowel,
    /// A consonant with a terminating vowel
    ConsonantWithTerminator,
    /// A consonant followed by hasant
    ConsonantWithHasant,
    /// A conjunct (multiple consonants joined with hasant)
    Conjunct,
    /// A conjunct with a vowel modifier
    ConjunctWithVowel,
    /// A conjunct with a terminating vowel
    ConjunctWithTerminator,
    /// A reph (র্) over a consonant
    RephOverConsonant,
    /// A reph over a consonant with a vowel
    RephOverConsonantWithVowel,
    /// A reph over a consonant with a terminator
    RephOverConsonantWithTerminator,
    /// A special form (e.g., reph, ya-phala, etc.)
    SpecialForm,
    /// A numeral
    Numeral,
    /// A symbol or punctuation
    Symbol,
    /// Unknown unit
    Unknown,
}

/// Tokenizer for processing input text
pub struct Tokenizer {
    /// Unified matcher using longest deterministic prefix matching.
    phonetic_patterns: &'static PatternTrie,
}

impl Tokenizer {
    /// Create a new tokenizer with default configuration
    pub fn new() -> Self {
        Tokenizer {
            phonetic_patterns: phonetic_pattern_trie_static(),
        }
    }

    /// Tokenize input text into words and other tokens
    pub fn tokenize_text(&self, text: &str) -> Vec<Token> {
        scanner::tokenize_text(text)
    }

    /// Tokenize a word into phonetic units for Bengali transliteration
    pub fn tokenize_word(&self, word: &str) -> Vec<PhoneticUnit> {
        let mut units = Vec::new();
        self.tokenize_word_into(word, &mut units);
        units
    }

    pub(crate) fn tokenize_word_into(&self, word: &str, units: &mut Vec<PhoneticUnit>) {
        units.clear();
        // Process the word character by character
        let mut _i = 0;

        let (processed_word, trailing_diacritics) = split_trailing_diacritics(word);

        // Special case for standalone diacritics
        if processed_word.is_empty() && !trailing_diacritics.is_empty() {
            append_trailing_diacritics(units, trailing_diacritics);
            return;
        }

        let mut scan_hints = WordScanHints::default();

        // Process the base word without diacritics
        while _i < processed_word.len() {
            if let Some(rule_match) = self.phonetic_patterns.match_at(processed_word, _i) {
                let unit = PhoneticUnit {
                    text: rule_match.text.to_string(),
                    unit_type: rule_match.unit_type,
                    position: _i,
                };
                scan_hints.observe_unit(&unit, units.last());
                units.push(unit);
                _i += rule_match.text.len();
                continue;
            }

            // If no pattern matched, treat as unknown and advance by one character
            if _i < processed_word.len() {
                // Find the length of one UTF-8 character
                let char_len = processed_word[_i..]
                    .chars()
                    .next()
                    .map_or(1, |c| c.len_utf8());

                let unknown_text = &processed_word[_i.._i + char_len];
                scan_hints.observe_unknown_text(unknown_text, processed_word, _i);

                units.push(PhoneticUnit {
                    text: unknown_text.to_string(),
                    unit_type: unmatched_unit_type(unknown_text),
                    position: _i,
                });
                _i += char_len;
            }
        }

        // Post-processing to identify conjuncts and other complex forms
        identify_complex_forms(units, scan_hints);

        append_trailing_diacritics(units, trailing_diacritics);
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

fn unmatched_unit_type(text: &str) -> PhoneticUnitType {
    if text.len() == 1 && text.as_bytes()[0].is_ascii_digit() {
        PhoneticUnitType::Numeral
    } else {
        PhoneticUnitType::Unknown
    }
}

fn move_unit(units: &mut [PhoneticUnit], read: usize, write: usize) {
    if write != read {
        units[write] = PhoneticUnit {
            text: std::mem::take(&mut units[read].text),
            unit_type: units[read].unit_type,
            position: units[read].position,
        };
    }
}

fn reph_base_part(unit: &PhoneticUnit) -> Option<&str> {
    if unit.unit_type == PhoneticUnitType::RephOverConsonant {
        unit.text.strip_prefix("rr").filter(|part| !part.is_empty())
    } else {
        None
    }
}
