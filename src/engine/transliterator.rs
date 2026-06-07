//! Main transliteration engine for Roman to Bengali conversion.
//!
//! This module contains the core logic for transliterating Roman text to Bengali.
//!
//! For detailed implementation rules, see docs/simplified_rules.md

use super::sanitizer::{SanitizeResult, Sanitizer};
use super::tokenizer::{PhoneticUnit, PhoneticUnitType, Token, TokenType, Tokenizer};
use crate::definitions::{
    conjuncts::{conjuncts, ConjunctDefinitions},
    consonant_value, diacritic_value,
    numerals::bengali_digit,
    symbol_value, vowel_value,
};
use std::borrow::Cow;

mod boundary;
mod components;
mod parts;

use boundary::{
    ends_with_khanda_ta_base_signal, is_decimal_separator, is_decimal_separator_at, next_char,
    starts_with_cluster, TokenNumberBoundary,
};
use components::{
    split_conjunct_component_vowel, split_consonant_vowel, split_reph_consonant_vowel,
};
use parts::ConjunctParts;

/// Main transliterator that performs the Roman to Bengali conversion
pub struct Transliterator {
    // Lookup tables for conversion
    conjuncts: &'static ConjunctDefinitions,

    // Input sanitizer
    sanitizer: Sanitizer,

    // Tokenizer
    tokenizer: Tokenizer,
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

    fn conjunct_component(&self, part: &str) -> Option<&'static str> {
        match part {
            "y" | "Y" => Some("য"),
            "w" => Some("ব"),
            _ => consonant_value(part),
        }
    }

    fn render_conjunct_parts(&self, parts: &[&str]) -> Option<Cow<'static, str>> {
        if parts.len() < 2 {
            return None;
        }

        if let Some(mapped) = self.conjuncts.create_conjunct_from_parts(parts) {
            return Some(Cow::Borrowed(mapped));
        }

        if parts.first() == Some(&"rr") {
            let tail = self.render_conjunct_parts(&parts[1..])?;
            let hasant = diacritic_value(",,").unwrap_or("্");
            let mut rendered = String::from("র");
            rendered.push_str(hasant);
            rendered.push_str(tail.as_ref());
            return Some(Cow::Owned(rendered));
        }

        let hasant = diacritic_value(",,").unwrap_or("্");
        let mut rendered = String::new();

        for (index, part) in parts.iter().enumerate() {
            rendered.push_str(self.conjunct_component(part)?);
            if index < parts.len() - 1 {
                rendered.push_str(hasant);
            }
        }

        Some(Cow::Owned(rendered))
    }

    fn append_dependent_vowel(&self, output: &mut String, vowel_key: &str) -> bool {
        if let Some(vowel) = vowel_value(vowel_key) {
            if let Some(dependent) = &vowel.dependent {
                output.push_str(dependent);
            }
            true
        } else {
            false
        }
    }

    fn should_suppress_visible_a(&self, vowel_key: &str, following_units: &[PhoneticUnit]) -> bool {
        vowel_key == "a" && starts_with_cluster(following_units)
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
        let mut result = String::with_capacity(estimated_text_render_capacity(text));
        let mut current_word_start = None;
        let mut current_word_end = 0;
        let mut previous_boundary = TokenNumberBoundary::default();
        let mut phonetic_units = Vec::new();

        let mut i = 0;
        while i < text.len() {
            let character = text[i..].chars().next().unwrap();
            let char_len = character.len_utf8();

            if character == '^' || character == ':' {
                if current_word_start.is_none() {
                    current_word_start = Some(i);
                }
                current_word_end = i + char_len;
                i += char_len;
                continue;
            }

            if character == '`' && next_char(text, i, char_len) == Some('`') {
                if let Some(start) = current_word_start {
                    if ends_with_khanda_ta_base_signal(&text[start..current_word_end]) {
                        i += 2;
                        current_word_end = i;
                        continue;
                    }
                }
            }

            if character == ',' && next_char(text, i, char_len) == Some(',') {
                if current_word_start.is_none() {
                    current_word_start = Some(i);
                }
                i += 2;
                current_word_end = i;
                continue;
            }

            if character.is_whitespace() {
                self.flush_current_word(
                    &mut result,
                    text,
                    &mut current_word_start,
                    current_word_end,
                    &mut previous_boundary,
                    &mut phonetic_units,
                );
                result.push(character);
                previous_boundary = TokenNumberBoundary::default();
            } else if character.is_ascii_punctuation() {
                let current_word = current_word_start.map(|start| &text[start..current_word_end]);
                let is_decimal =
                    is_decimal_separator_at(text, i, char_len, current_word, previous_boundary);
                self.flush_current_word(
                    &mut result,
                    text,
                    &mut current_word_start,
                    current_word_end,
                    &mut previous_boundary,
                    &mut phonetic_units,
                );

                if is_decimal {
                    result.push('.');
                } else if let Some(bengali_symbol) = symbol_value(&text[i..i + char_len]) {
                    result.push_str(bengali_symbol);
                } else {
                    result.push(character);
                }
                previous_boundary = TokenNumberBoundary::default();
            } else if !character.is_alphanumeric() {
                self.flush_current_word(
                    &mut result,
                    text,
                    &mut current_word_start,
                    current_word_end,
                    &mut previous_boundary,
                    &mut phonetic_units,
                );

                let symbol_text = &text[i..i + char_len];
                if let Some(bengali_symbol) = symbol_value(symbol_text) {
                    result.push_str(bengali_symbol);
                } else {
                    result.push_str(symbol_text);
                }
                previous_boundary = TokenNumberBoundary::default();
            } else {
                if current_word_start.is_none() {
                    current_word_start = Some(i);
                }
                current_word_end = i + char_len;
            }

            i += char_len;
        }

        self.flush_current_word(
            &mut result,
            text,
            &mut current_word_start,
            current_word_end,
            &mut previous_boundary,
            &mut phonetic_units,
        );

        result
    }

    fn flush_current_word(
        &self,
        result: &mut String,
        text: &str,
        current_word_start: &mut Option<usize>,
        current_word_end: usize,
        previous_boundary: &mut TokenNumberBoundary,
        phonetic_units: &mut Vec<PhoneticUnit>,
    ) {
        let Some(start) = current_word_start.take() else {
            return;
        };
        let current_word = &text[start..current_word_end];

        *previous_boundary = TokenNumberBoundary::from_word(current_word);

        if current_word.chars().all(|character| character.is_numeric()) {
            self.render_number_token(result, current_word);
        } else {
            self.transliterate_word_units_into(result, current_word, phonetic_units);
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
        if self.sanitizer.is_valid(text) {
            self.render_text(text)
        } else {
            // Keep the total `transliterate` API side-effect free. Callers
            // that need details can use `sanitize` before transliterating.
            text.to_string()
        }
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
        // Clean the input by removing invalid characters
        let cleaned = self.sanitizer.clean(text);

        // Process the cleaned text using the tokenizer
        let tokens = self.tokenizer.tokenize_text(&cleaned);

        self.render_tokens(&tokens)
    }

    fn transliterate_word_units_into(
        &self,
        result: &mut String,
        word: &str,
        phonetic_units: &mut Vec<PhoneticUnit>,
    ) {
        self.tokenizer.tokenize_word_into(word, phonetic_units);

        let mut previous_unit_accepts_dependent_vowel = false;

        let unit_count = phonetic_units.len();
        for (unit_index, unit) in phonetic_units.iter().enumerate() {
            let is_last_unit = unit_index + 1 == unit_count;
            let following_units = &phonetic_units[unit_index + 1..];

            match unit.unit_type {
                PhoneticUnitType::Consonant => {
                    if let Some(bengali_consonant) = consonant_value(unit.text.as_str()) {
                        result.push_str(bengali_consonant);
                        previous_unit_accepts_dependent_vowel = true;
                    } else {
                        // Fallback: keep original text
                        result.push_str(&unit.text);
                        previous_unit_accepts_dependent_vowel = false;
                    }
                }
                PhoneticUnitType::Vowel => {
                    if let Some(vowel) = vowel_value(unit.text.as_str()) {
                        if previous_unit_accepts_dependent_vowel {
                            // If preceded by a consonant, use dependent form if available
                            if let Some(dependent) = &vowel.dependent {
                                result.push_str(dependent);
                            } else {
                                // If no dependent form exists, use independent as fallback
                                result.push_str(vowel.independent);
                            }
                        } else {
                            // Use the independent form for standalone vowels
                            result.push_str(vowel.independent);
                        }
                        previous_unit_accepts_dependent_vowel = false;
                    } else {
                        // Fallback: keep original text
                        result.push_str(&unit.text);
                        previous_unit_accepts_dependent_vowel = false;
                    }
                }
                PhoneticUnitType::TerminatingVowel => {
                    if let Some(vowel) = vowel_value(unit.text.as_str()) {
                        if previous_unit_accepts_dependent_vowel {
                            // If preceded by a consonant, use dependent form if available
                            if let Some(dependent) = &vowel.dependent {
                                result.push_str(dependent);
                            } else {
                                // If no dependent form exists, use independent as fallback
                                result.push_str(vowel.independent);
                            }
                        } else {
                            // Use the independent form for standalone terminating vowels
                            result.push_str(vowel.independent);
                        }
                        previous_unit_accepts_dependent_vowel = false;
                    } else {
                        // Fallback: keep original text
                        result.push_str(&unit.text);
                        previous_unit_accepts_dependent_vowel = false;
                    }
                }
                PhoneticUnitType::ConsonantWithVowel => {
                    if let Some((consonant_part, vowel_part)) = split_consonant_vowel(&unit.text) {
                        if let Some(bengali_consonant) = consonant_value(consonant_part) {
                            result.push_str(bengali_consonant);
                            if let Some(vowel) = vowel_value(vowel_part) {
                                if self.should_suppress_visible_a(vowel_part, following_units) {
                                    previous_unit_accepts_dependent_vowel = true;
                                    continue;
                                }

                                if let Some(dependent) = &vowel.dependent {
                                    result.push_str(dependent);
                                } else {
                                    result.push_str(vowel.independent);
                                }
                            } else {
                                result.push_str(vowel_part);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else if let Some(bengali_consonant) = consonant_value(unit.text.as_str()) {
                        result.push_str(bengali_consonant);
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::ConsonantWithTerminator => {
                    // Process consonant with terminating vowel (like o, O)
                    // For consonants like "th" we need to check if they exist in our consonant map
                    // Extract the consonant and terminator parts
                    if let Some((consonant_part, terminator_part)) =
                        split_consonant_vowel(&unit.text)
                    {
                        if let Some(bengali_consonant) = consonant_value(consonant_part) {
                            // Add the consonant
                            result.push_str(bengali_consonant);

                            // Handle the terminator - if it's 'o', it's the inherent vowel in Bengali
                            // and doesn't need a separate symbol
                            if terminator_part != "o" {
                                if let Some(vowel) = vowel_value(terminator_part) {
                                    if let Some(dependent) = &vowel.dependent {
                                        result.push_str(dependent);
                                    } else {
                                        // Fallback to independent form if dependent not available
                                        result.push_str(vowel.independent);
                                    }
                                } else {
                                    // Terminator part not recognized, just append it
                                    result.push_str(terminator_part);
                                }
                            }
                        } else {
                            // Consonant not recognized, just use the original text
                            result.push_str(&unit.text);
                        }
                    } else {
                        // No vowel found, treat the whole thing as a consonant
                        if let Some(bengali_consonant) = consonant_value(unit.text.as_str()) {
                            result.push_str(bengali_consonant);
                        } else {
                            // Fallback: keep original text
                            result.push_str(&unit.text);
                        }
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::ConsonantWithHasant => {
                    // Explicit hasant marker. It may attach to a preceding
                    // consonant or stand alone as a deliberate virama signal.
                    if unit.text == ",," {
                        let hasant = diacritic_value(",,").unwrap_or("্");
                        result.push_str(hasant);
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::Conjunct => {
                    let parts = ConjunctParts::from_text(&unit.text);

                    if let Some(rendered) = self.render_conjunct_parts(parts.as_slice()) {
                        result.push_str(&rendered);
                    } else {
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::ConjunctWithVowel => {
                    let mut parts = ConjunctParts::from_text(&unit.text);

                    if parts.len() >= 2 {
                        let last_part = parts.last().expect("parts length checked");
                        if let Some((last_consonant, vowel_part)) =
                            split_conjunct_component_vowel(last_part)
                        {
                            parts.replace_last(last_consonant);

                            if let Some(rendered) = self.render_conjunct_parts(parts.as_slice()) {
                                result.push_str(&rendered);
                                if !matches!(last_consonant, "y" | "Y" | "w")
                                    && self.should_suppress_visible_a(vowel_part, following_units)
                                {
                                    previous_unit_accepts_dependent_vowel = true;
                                } else if !self.append_dependent_vowel(result, vowel_part) {
                                    result.push_str(vowel_part);
                                }
                            } else {
                                result.push_str(&unit.text);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::ConjunctWithTerminator => {
                    let mut parts = ConjunctParts::from_text(&unit.text);

                    if parts.len() >= 2 {
                        let last_part = parts.last().expect("parts length checked");
                        if let Some((last_consonant, terminator_part)) =
                            split_conjunct_component_vowel(last_part)
                        {
                            parts.replace_last(last_consonant);

                            if let Some(rendered) = self.render_conjunct_parts(parts.as_slice()) {
                                result.push_str(&rendered);
                                if terminator_part == "o" {
                                    if is_last_unit && matches!(last_consonant, "y" | "Y" | "w") {
                                        self.append_dependent_vowel(result, "O");
                                    }
                                } else if !self.append_dependent_vowel(result, terminator_part) {
                                    result.push_str(terminator_part);
                                }
                            } else {
                                result.push_str(&unit.text);
                            }
                        } else if let Some(rendered) = self.render_conjunct_parts(parts.as_slice())
                        {
                            result.push_str(&rendered);
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::RephOverConsonant => {
                    if let Some(mapped) = self.conjuncts.create_conjunct(&unit.text) {
                        result.push_str(mapped);
                    } else {
                        // Process reph over consonant (র্ + consonant)
                        // Extract the consonant part (after "rr")
                        let consonant_text = &unit.text[2..]; // Skip the "rr" prefix

                        if let Some(bengali_consonant) = consonant_value(consonant_text) {
                            // Create reph + consonant (reph comes before consonant in Bengali)
                            // In Bengali, reph is represented as র + hasant (্)
                            let reph = "র্"; // Fixed Bengali reph
                            result.push_str(reph);
                            result.push_str(bengali_consonant);
                        } else {
                            // Fallback: keep original text
                            result.push_str(&unit.text);
                        }
                    }
                }
                PhoneticUnitType::RephOverConsonantWithVowel => {
                    // Process reph over consonant with vowel (র্ + consonant + vowel)
                    // This is a complex form that needs to be processed properly
                    // For example, "rrka" should become "র্ক" + vowel sign

                    if let Some((consonant_part, vowel_part)) =
                        split_reph_consonant_vowel(&unit.text)
                    {
                        let reph_parts = ["rr", consonant_part];
                        if let Some(mapped) = self.conjuncts.create_conjunct_from_parts(&reph_parts)
                        {
                            result.push_str(mapped);
                            if !self.append_dependent_vowel(result, vowel_part) {
                                result.push_str(vowel_part);
                            }
                        } else if let (Some(bengali_consonant), Some(vowel)) =
                            (consonant_value(consonant_part), vowel_value(vowel_part))
                        {
                            // Create reph + consonant (reph comes before consonant in Bengali)
                            // In Bengali, reph is represented as র + hasant (্)
                            let reph = "র্"; // Fixed Bengali reph
                            result.push_str(reph);
                            result.push_str(bengali_consonant);

                            // Handle Option<&str> correctly for dependent vowel
                            if let Some(dependent_vowel) = &vowel.dependent {
                                result.push_str(dependent_vowel);
                            } else {
                                // If no dependent form exists, use independent as fallback
                                result.push_str(vowel.independent);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        // Reph body not recognized
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::RephOverConsonantWithTerminator => {
                    // Process reph over consonant with terminator (র্ + consonant + o)
                    // Similar to RephOverConsonantWithVowel but with terminator vowel

                    if let Some((consonant_part, terminator_part)) =
                        split_reph_consonant_vowel(&unit.text)
                    {
                        let reph_parts = ["rr", consonant_part];
                        if let Some(mapped) = self.conjuncts.create_conjunct_from_parts(&reph_parts)
                        {
                            result.push_str(mapped);

                            if !terminator_part.is_empty()
                                && terminator_part != "o"
                                && !self.append_dependent_vowel(result, terminator_part)
                            {
                                result.push_str(terminator_part);
                            }
                        } else if let Some(bengali_consonant) = consonant_value(consonant_part) {
                            // Create reph + consonant
                            let reph = "র্"; // Fixed Bengali reph
                            result.push_str(reph);
                            result.push_str(bengali_consonant);

                            // The explicit `o` terminator marks the inherent vowel and
                            // should not render as an independent অ after a consonant.
                            if !terminator_part.is_empty() && terminator_part != "o" {
                                if let Some(vowel) = vowel_value(terminator_part) {
                                    if let Some(dependent) = &vowel.dependent {
                                        result.push_str(dependent);
                                    } else {
                                        result.push_str(vowel.independent);
                                    }
                                }
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        // Reph body not recognized
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::SpecialForm => {
                    // Special forms with proper text field handling
                    if unit.text == "rr" {
                        // Standalone reph is র্
                        result.push_str("র্");
                    } else if unit.text == "^" {
                        // Standalone Chandrabindu
                        if let Some(chandrabindu) = diacritic_value("^") {
                            result.push_str(chandrabindu);
                        } else {
                            result.push('ঁ');
                        }
                    } else if unit.text == ":" {
                        // Standalone Visarga - now handled directly here
                        if let Some(visarga) = diacritic_value(":") {
                            result.push_str(visarga);
                        } else {
                            result.push('ঃ');
                        }
                    } else if matches!(unit.text.as_str(), "t``" | "T``") {
                        // Handle Khanda Ta (special form of ta)
                        let khanda_ta = diacritic_value(unit.text.as_str()).unwrap_or("ৎ");
                        result.push_str(khanda_ta);
                    } else if matches!(unit.text.as_str(), "ng" | "M") {
                        // Handle anusvara (ং)
                        if let Some(anusvara) = diacritic_value(unit.text.as_str()) {
                            result.push_str(anusvara);
                        } else {
                            result.push('ং');
                        }
                    } else {
                        // Fallback: keep original text
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::Numeral => {
                    self.render_number_token(result, &unit.text);
                }
                PhoneticUnitType::Symbol => {
                    // Convert to Bengali symbol if applicable
                    if let Some(bengali_symbol) = symbol_value(unit.text.as_str()) {
                        result.push_str(bengali_symbol);
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::Unknown => {
                    result.push_str(&unit.text);
                    previous_unit_accepts_dependent_vowel = false;
                }
            }
        }
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
