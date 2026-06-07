//! Main transliteration engine for Roman to Bengali conversion.
//!
//! This module contains the core logic for transliterating Roman text to Bengali.
//!
//! For detailed implementation rules, see data/rules/simplified_rules.md

use super::sanitizer::{SanitizeResult, Sanitizer};
use super::text_boundary::{
    is_explicit_hasant_signal_at, is_khanda_ta_suffix_signal_at, is_phonetic_mark_signal,
};
use super::tokenizer::{PhoneticUnit, Token, TokenType, Tokenizer};
use crate::definitions::{
    conjuncts::{conjuncts, ConjunctDefinitions},
    numerals::bengali_digit,
    symbol_value,
};

mod boundary;
mod components;
mod parts;
mod word;

use boundary::{is_decimal_separator, is_decimal_separator_at, TokenNumberBoundary};

/// Main transliterator that performs the Roman to Bengali conversion
pub struct Transliterator {
    // Lookup tables for conversion
    conjuncts: &'static ConjunctDefinitions,

    // Input sanitizer
    sanitizer: Sanitizer,

    // Tokenizer
    tokenizer: Tokenizer,
}

struct TextRenderState {
    current_word_start: Option<usize>,
    current_word_end: usize,
    current_word_is_number: bool,
    current_word_ends_with_number: bool,
    previous_boundary: TokenNumberBoundary,
    phonetic_units: Vec<PhoneticUnit>,
}

impl TextRenderState {
    fn new() -> Self {
        Self {
            current_word_start: None,
            current_word_end: 0,
            current_word_is_number: true,
            current_word_ends_with_number: false,
            previous_boundary: TokenNumberBoundary::default(),
            phonetic_units: Vec::new(),
        }
    }
}

impl Transliterator {
    /// Create a new transliterator with default configuration
    pub fn new() -> Self {
        Transliterator {
            // Lookup tables for conversion
            conjuncts: conjuncts(),

            // Input sanitizer
            sanitizer: Sanitizer::default(),

            // Tokenizer
            tokenizer: Tokenizer::default(),
        }
    }

    fn render_tokens(&self, tokens: &[Token]) -> String {
        let mut result = String::with_capacity(estimated_render_capacity(tokens));
        let mut phonetic_units = Vec::new();

        for index in 0..tokens.len() {
            self.render_token_at_into(&mut result, tokens, index, &mut phonetic_units);
        }

        result
    }

    fn render_token_at_into(
        &self,
        result: &mut String,
        tokens: &[Token],
        index: usize,
        phonetic_units: &mut Vec<PhoneticUnit>,
    ) {
        let token = &tokens[index];

        match token.token_type {
            TokenType::Word => {
                self.transliterate_word_units_into(result, &token.content, phonetic_units);
            }
            TokenType::Whitespace => {
                result.push_str(&token.content);
            }
            TokenType::Punctuation | TokenType::Symbol => {
                if is_decimal_separator(tokens, index) {
                    result.push('.');
                } else if let Some(bengali_symbol) = symbol_value(token.content.as_str()) {
                    result.push_str(bengali_symbol);
                } else {
                    result.push_str(&token.content);
                }
            }
            TokenType::Number => {
                self.render_number_token(result, &token.content);
            }
        }
    }

    fn render_text(&self, text: &str) -> String {
        self.render_text_inner::<false>(text)
            .expect("unchecked render path should not reject input")
    }

    fn render_text_checked(&self, text: &str) -> Option<String> {
        self.render_text_inner::<true>(text)
    }

    fn render_text_inner<const CHECK_INPUT: bool>(&self, text: &str) -> Option<String> {
        let mut result = String::with_capacity(estimated_text_render_capacity(text));
        let mut state = TextRenderState::new();

        let mut i = 0;
        while i < text.len() {
            let character = text[i..].chars().next().unwrap();
            let char_len = character.len_utf8();

            if CHECK_INPUT && !self.sanitizer.is_allowed_char(character) {
                return None;
            }

            if is_phonetic_mark_signal(character) {
                if state.current_word_start.is_none() {
                    state.current_word_start = Some(i);
                }
                state.current_word_is_number = false;
                state.current_word_ends_with_number = false;
                state.current_word_end = i + char_len;
                i += char_len;
                continue;
            }

            if let Some(start) = state.current_word_start {
                if is_khanda_ta_suffix_signal_at(
                    character,
                    text,
                    i,
                    &text[start..state.current_word_end],
                ) {
                    state.current_word_is_number = false;
                    state.current_word_ends_with_number = false;
                    i += 2;
                    state.current_word_end = i;
                    continue;
                }
            }

            if is_explicit_hasant_signal_at(character, text, i) {
                if state.current_word_start.is_none() {
                    state.current_word_start = Some(i);
                }
                state.current_word_is_number = false;
                state.current_word_ends_with_number = false;
                i += 2;
                state.current_word_end = i;
                continue;
            }

            if character.is_whitespace() {
                self.flush_current_word(&mut result, text, &mut state);
                result.push(character);
                state.previous_boundary = TokenNumberBoundary::default();
            } else if character.is_ascii_punctuation() {
                let current_word = state.current_word_start.map(|_| {
                    TokenNumberBoundary::from_number_state(state.current_word_ends_with_number)
                });
                let is_decimal = is_decimal_separator_at(
                    text,
                    i,
                    char_len,
                    current_word,
                    state.previous_boundary,
                );
                self.flush_current_word(&mut result, text, &mut state);

                if is_decimal {
                    result.push('.');
                } else if let Some(bengali_symbol) = symbol_value(&text[i..i + char_len]) {
                    result.push_str(bengali_symbol);
                } else {
                    result.push(character);
                }
                state.previous_boundary = TokenNumberBoundary::default();
            } else if !character.is_alphanumeric() {
                self.flush_current_word(&mut result, text, &mut state);

                let symbol_text = &text[i..i + char_len];
                if let Some(bengali_symbol) = symbol_value(symbol_text) {
                    result.push_str(bengali_symbol);
                } else {
                    result.push_str(symbol_text);
                }
                state.previous_boundary = TokenNumberBoundary::default();
            } else {
                if state.current_word_start.is_none() {
                    state.current_word_start = Some(i);
                    state.current_word_is_number = true;
                }
                let is_number = character.is_numeric();
                state.current_word_is_number &= is_number;
                state.current_word_ends_with_number = is_number;
                state.current_word_end = i + char_len;
            }

            i += char_len;
        }

        self.flush_current_word(&mut result, text, &mut state);

        Some(result)
    }

    fn flush_current_word(&self, result: &mut String, text: &str, state: &mut TextRenderState) {
        let Some(start) = state.current_word_start.take() else {
            state.current_word_is_number = true;
            state.current_word_ends_with_number = false;
            return;
        };
        let current_word = &text[start..state.current_word_end];
        let is_number = state.current_word_is_number;
        let ends_with_number = state.current_word_ends_with_number;
        state.current_word_is_number = true;
        state.current_word_ends_with_number = false;

        state.previous_boundary = TokenNumberBoundary::from_number_state(ends_with_number);

        if is_number {
            self.render_number_token(result, current_word);
        } else {
            self.transliterate_word_units_into(result, current_word, &mut state.phonetic_units);
        }
    }

    /// Render already-tokenized input to Bengali.
    ///
    /// Callers that need phase-level profiling can sanitize and tokenize once,
    /// then measure only this rendering stage instead of calling the full
    /// `transliterate` pipeline again.
    pub fn transliterate_tokens(&self, tokens: &[Token]) -> String {
        self.render_tokens(tokens)
    }

    /// Render a single token with access to its already-tokenized neighbors.
    ///
    /// This is intended for debug/verbose views that need per-token output
    /// without losing context-sensitive rules such as decimal punctuation.
    pub fn transliterate_token_at(&self, tokens: &[Token], index: usize) -> Option<String> {
        if index >= tokens.len() {
            return None;
        }

        let token = &tokens[index];
        let mut result = String::with_capacity(token.content.len().saturating_mul(3));
        let mut phonetic_units = Vec::new();
        self.render_token_at_into(&mut result, tokens, index, &mut phonetic_units);
        Some(result)
    }

    fn render_number_token(&self, result: &mut String, content: &str) {
        for digit in content.chars() {
            if let Some(mapped) = bengali_digit(digit) {
                result.push_str(mapped);
            } else {
                result.push(digit);
            }
        }
    }

    /// Transliterate Roman text to Bengali
    pub fn transliterate(&self, text: &str) -> String {
        // Keep the total `transliterate` API side-effect free. Callers that
        // need details can use `sanitize` before transliterating. Validation is
        // fused into rendering so the common valid path does not scan twice.
        self.render_text_checked(text)
            .unwrap_or_else(|| text.to_string())
    }

    /// Tokenize the input text into words and other tokens
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        self.tokenizer.tokenize_text(text)
    }

    /// Tokenize a word into phonetic units
    pub fn tokenize_phonetic(&self, word: &str) -> Vec<PhoneticUnit> {
        self.tokenizer.tokenize_word(word)
    }

    /// Sanitize the input text, ensuring it contains only allowed characters
    pub fn sanitize(&self, text: &str) -> SanitizeResult {
        self.sanitizer.sanitize(text)
    }

    /// Transliterate Roman text to Bengali, cleaning invalid characters instead of returning an error
    pub fn transliterate_lenient(&self, text: &str) -> String {
        let cleaned = self.sanitizer.clean(text);
        self.render_text(&cleaned)
    }
}

impl Default for Transliterator {
    fn default() -> Self {
        Self::new()
    }
}

fn estimated_render_capacity(tokens: &[Token]) -> usize {
    tokens.iter().fold(0usize, |capacity, token| {
        capacity.saturating_add(token.content.len().saturating_mul(3))
    })
}

fn estimated_text_render_capacity(text: &str) -> usize {
    text.len().saturating_mul(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_render_matches_tokenized_render_for_boundaries() {
        let transliterator = Transliterator::new();

        for input in [
            "12.34 12.34.",
            "k12.34 a1.b2",
            "12.34.56 12 .34 12..34 1.a2",
            "k,,k t`` T`` :^",
            "rrkSh rrk,,Sh k,,w k,,y",
            "t`` T`` kt``a T``o",
            "^ami :shokal shesh^",
            "rZyab rrYa Zya kZya rZga",
            "boi bou kOI kOU kOko kok",
            "ami\u{00a0}bangla\u{2003}lekhi ১২.৩৪",
            "আমি kA লিখি।",
        ] {
            let tokens = transliterator.tokenize(input);
            assert_eq!(
                transliterator.transliterate(input),
                transliterator.transliterate_tokens(&tokens),
                "direct render diverged from tokenized render for {input}"
            );
        }
    }
}
