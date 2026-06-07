//! Tokenizer for the Obadh Engine
//!
//! This module provides functionality to tokenize input text into words
//! and letters/phonemes for processing by the transliteration engine.

use crate::definitions::{conjuncts, consonant_value, vowel_value};

mod patterns;

use patterns::{phonetic_pattern_trie_static, PatternTrie};

const MAX_INLINE_EXPLICIT_PARTS: usize = 8;

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
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut current_position = 0;

        let mut i = 0;
        while i < text.len() {
            // Get the current character
            let c = text[i..].chars().next().unwrap();
            let char_len = c.len_utf8();

            // Chandrabindu and visarga are phonetic mark signals. Keep them
            // inside word tokens even when they appear standalone so
            // tokenize_word applies the same deterministic rendering path.
            if c == '^' || c == ':' {
                if current_word.is_empty() {
                    current_position = i;
                }
                current_word.push(c);
                i += char_len;
                continue;
            }

            // Special case: Check for khanda-ta notation that should attach to
            // the previous word.
            if !current_word.is_empty() && c == '`' {
                // Special case for Khanda Ta (t`` / T``)
                if c == '`'
                    && next_char(text, i, char_len) == Some('`')
                    && ends_with_khanda_ta_base_signal(&current_word)
                {
                    // Add the `` to mark it as Khanda Ta
                    current_word.push_str("``");
                    i += 2; // Skip both backticks
                    continue;
                }
            }

            // Special case: Check for hasanta sequence (,,)
            if c == ',' && next_char(text, i, char_len) == Some(',') {
                if current_word.is_empty() {
                    current_position = i;
                }

                current_word.push_str(",,");
                i += 2; // Skip both commas
                continue;
            }

            if c.is_whitespace() {
                // Add the current word if any
                push_current_word_token(&mut current_word, current_position, &mut tokens);

                // Add the whitespace as a token
                tokens.push(Token {
                    content: c.to_string(),
                    token_type: TokenType::Whitespace,
                    position: i,
                });

                current_position = i + char_len;
            } else if c.is_ascii_punctuation() {
                // Add the current word if any
                push_current_word_token(&mut current_word, current_position, &mut tokens);

                // Add the punctuation as a token
                tokens.push(Token {
                    content: c.to_string(),
                    token_type: TokenType::Punctuation,
                    position: i,
                });

                current_position = i + char_len;
            } else if !c.is_alphanumeric() {
                // Special symbol - add the current word if any
                push_current_word_token(&mut current_word, current_position, &mut tokens);

                // Add the symbol as a token
                tokens.push(Token {
                    content: c.to_string(),
                    token_type: TokenType::Symbol,
                    position: i,
                });

                current_position = i + char_len;
            } else {
                // If we have an empty current word, update the position
                if current_word.is_empty() {
                    current_position = i;
                }
                // Add the character to the current word
                current_word.push(c);
            }

            i += char_len;
        }

        // Add any remaining word
        push_current_word_token(&mut current_word, current_position, &mut tokens);

        tokens
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

        let mut has_long_iya_marker_candidate = false;

        // Process the base word without diacritics
        while _i < processed_word.len() {
            if let Some(rule_match) = self.phonetic_patterns.match_at(processed_word, _i) {
                units.push(PhoneticUnit {
                    text: rule_match.text.to_string(),
                    unit_type: rule_match.unit_type,
                    position: _i,
                });
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
                if unknown_text == "w" && is_long_iya_marker_at(processed_word, _i) {
                    has_long_iya_marker_candidate = true;
                }

                units.push(PhoneticUnit {
                    text: unknown_text.to_string(),
                    unit_type: unmatched_unit_type(unknown_text),
                    position: _i,
                });
                _i += char_len;
            }
        }

        // Post-processing to identify conjuncts and other complex forms
        self.identify_complex_forms(units, has_long_iya_marker_candidate);

        append_trailing_diacritics(units, trailing_diacritics);
    }

    /// Identify complex phonetic forms like conjuncts and consonants with vowel modifiers
    fn identify_complex_forms(
        &self,
        units: &mut Vec<PhoneticUnit>,
        has_long_iya_marker_candidate: bool,
    ) {
        // Get a reference to the conjunct definitions
        let conjunct_defs = conjuncts();

        // First pass: Handle special "rr" cases
        // - "rr" as vocalic R vowel
        // - "rr" + consonant as reph
        normalize_redundant_reph_hasant(units);

        normalize_reph_and_vocalic_r(units);

        normalize_velar_nasal_conjunct_aliases(units);
        normalize_redundant_khanda_ta_hasant(units);

        // Collapse explicit hasant chains, e.g. n,,d,,r -> n,,d,,r.
        // Explicit hasant is a user command, so it is preserved even before
        // later valid-conjunct filtering and vowel attachment runs.
        collapse_explicit_hasant_chains(units, conjunct_defs);

        // Second pass: process contiguous consonant runs to form conjuncts.
        // Non-consonant units such as anusvara are boundaries, not blockers for
        // subsequent runs. Work directly on contiguous ranges to avoid
        // per-word segment/index allocations in the tokenizer hot path.
        let mut run_start = 0;
        while run_start < units.len() {
            while run_start < units.len() && !is_conjunct_run_component(&units[run_start]) {
                run_start += 1;
            }

            let mut run_end = run_start;
            while run_end < units.len() && is_conjunct_run_component(&units[run_end]) {
                run_end += 1;
            }

            form_conjuncts_in_range(units, run_start, run_end, conjunct_defs);
            run_start = run_end;
        }

        compact_units_and_attach_vowels(units);

        // Fourth pass: normalize the deliberate `iyw` long-iya signal.
        // If it consumes a marker, a following vowel may now be adjacent to
        // `y`/`Y`; run the ordinary attachment pass once more in that case.
        if has_long_iya_marker_candidate && normalize_iyw_long_iya_signal(units) {
            compact_units_and_attach_vowels(units);
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

fn push_current_word_token(word: &mut String, position: usize, tokens: &mut Vec<Token>) {
    if word.is_empty() {
        return;
    }

    let token_type = if word.chars().all(|c| c.is_numeric()) {
        TokenType::Number
    } else {
        TokenType::Word
    };

    let capacity = word.capacity();
    let content = std::mem::replace(word, String::with_capacity(capacity));

    tokens.push(Token {
        content,
        token_type,
        position,
    });
}

fn next_char(text: &str, byte_index: usize, current_char_len: usize) -> Option<char> {
    text.get(byte_index + current_char_len..)?.chars().next()
}

fn ends_with_khanda_ta_base_signal(text: &str) -> bool {
    text.chars()
        .next_back()
        .is_some_and(|c| matches!(c, 't' | 'T'))
}

fn is_long_iya_marker_at(text: &str, byte_index: usize) -> bool {
    byte_index > 0 && matches!(text.as_bytes().get(byte_index - 1), Some(b'y') | Some(b'Y'))
}

fn unmatched_unit_type(text: &str) -> PhoneticUnitType {
    if text.len() == 1 && text.as_bytes()[0].is_ascii_digit() {
        PhoneticUnitType::Numeral
    } else {
        PhoneticUnitType::Unknown
    }
}

fn compact_units_and_attach_vowels(units: &mut Vec<PhoneticUnit>) {
    let mut read = 0;
    let mut write = 0;

    while read < units.len() {
        while read < units.len() && units[read].text.is_empty() {
            read += 1;
        }
        if read >= units.len() {
            break;
        }

        if let Some(next) = next_non_empty_unit_index(units, read + 1) {
            if let Some(combined_type) =
                attached_vowel_unit_type(units[read].unit_type, units[next].unit_type)
            {
                move_unit(units, read, write);
                let vowel_text = std::mem::take(&mut units[next].text);
                units[write].text.push_str(&vowel_text);
                units[write].unit_type = combined_type;
                read = next + 1;
                write += 1;
                continue;
            }
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

fn attached_vowel_unit_type(
    base: PhoneticUnitType,
    vowel: PhoneticUnitType,
) -> Option<PhoneticUnitType> {
    match (base, vowel) {
        (PhoneticUnitType::Consonant, PhoneticUnitType::Vowel) => {
            Some(PhoneticUnitType::ConsonantWithVowel)
        }
        (PhoneticUnitType::Consonant, PhoneticUnitType::TerminatingVowel) => {
            Some(PhoneticUnitType::ConsonantWithTerminator)
        }
        (PhoneticUnitType::Conjunct, PhoneticUnitType::Vowel) => {
            Some(PhoneticUnitType::ConjunctWithVowel)
        }
        (PhoneticUnitType::Conjunct, PhoneticUnitType::TerminatingVowel) => {
            Some(PhoneticUnitType::ConjunctWithTerminator)
        }
        (PhoneticUnitType::RephOverConsonant, PhoneticUnitType::Vowel) => {
            Some(PhoneticUnitType::RephOverConsonantWithVowel)
        }
        (PhoneticUnitType::RephOverConsonant, PhoneticUnitType::TerminatingVowel) => {
            Some(PhoneticUnitType::RephOverConsonantWithTerminator)
        }
        _ => None,
    }
}

fn next_non_empty_unit_index(units: &[PhoneticUnit], start: usize) -> Option<usize> {
    units
        .iter()
        .enumerate()
        .skip(start)
        .find_map(|(index, unit)| (!unit.text.is_empty()).then_some(index))
}

fn form_conjuncts_in_range(
    units: &mut [PhoneticUnit],
    start: usize,
    end: usize,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) {
    if end.saturating_sub(start) <= 1 {
        return;
    }

    let mut i = start;

    while i < end {
        if units[i].text.is_empty() {
            i += 1;
            continue;
        }

        if let Some(length) = longest_conjunct_prefix_in_range(units, i, end, conjunct_defs) {
            let conjunct_text = conjunct_text_for_range(units, i, length);

            let position = units[i].position;
            units[i] = PhoneticUnit {
                text: conjunct_text,
                unit_type: PhoneticUnitType::Conjunct,
                position,
            };

            for unit in units.iter_mut().take(i + length).skip(i + 1) {
                unit.text.clear();
            }
        }

        i += 1;
    }
}

struct ExplicitHasantChain {
    end: usize,
    text: String,
    position: usize,
}

fn collapse_explicit_hasant_chains(
    units: &mut Vec<PhoneticUnit>,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) {
    let Some(first_match) = first_explicit_hasant_chain(units, conjunct_defs) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if let Some(chain) = explicit_hasant_chain_at(units, read, conjunct_defs) {
            units[write] = PhoneticUnit {
                text: chain.text,
                unit_type: PhoneticUnitType::Conjunct,
                position: chain.position,
            };
            read = chain.end;
            write += 1;
            continue;
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

fn first_explicit_hasant_chain(
    units: &[PhoneticUnit],
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> Option<usize> {
    (0..units.len().saturating_sub(2))
        .find(|&index| explicit_hasant_chain_at(units, index, conjunct_defs).is_some())
}

fn explicit_hasant_chain_at(
    units: &[PhoneticUnit],
    index: usize,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> Option<ExplicitHasantChain> {
    if index + 2 >= units.len()
        || units[index + 1].unit_type != PhoneticUnitType::ConsonantWithHasant
    {
        return None;
    }

    let mut parts = explicit_hasant_chain_start_parts(&units[index])?;
    let position = units[index].position;
    let mut next = index + 1;
    let mut consumed_hasant = false;

    while next < units.len() && units[next].unit_type == PhoneticUnitType::ConsonantWithHasant {
        if next + 1 >= units.len() {
            break;
        }

        if let Some(part) =
            explicit_hasant_chain_next_part(parts.as_slice(), &units[next + 1], conjunct_defs)
        {
            parts.push(part);
            consumed_hasant = true;
            next += 2;
        } else {
            break;
        }
    }

    if consumed_hasant
        && parts.len() >= 2
        && explicit_hasant_chain_is_renderable(parts.as_slice(), conjunct_defs)
    {
        return Some(ExplicitHasantChain {
            end: next,
            text: join_explicit_hasant_parts(parts.as_slice()),
            position,
        });
    }

    None
}

fn join_explicit_hasant_parts(parts: &[&str]) -> String {
    let separator_len = ",,".len() * parts.len().saturating_sub(1);
    let mut text =
        String::with_capacity(parts.iter().map(|part| part.len()).sum::<usize>() + separator_len);

    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            text.push_str(",,");
        }
        text.push_str(part);
    }

    text
}

struct BorrowedParts<'a> {
    inline: [&'a str; MAX_INLINE_EXPLICIT_PARTS],
    len: usize,
    overflow: Option<Vec<&'a str>>,
}

impl<'a> BorrowedParts<'a> {
    fn new() -> Self {
        Self {
            inline: [""; MAX_INLINE_EXPLICIT_PARTS],
            len: 0,
            overflow: None,
        }
    }

    fn from_one(first: &'a str) -> Self {
        let mut parts = Self::new();
        parts.push(first);
        parts
    }

    fn from_two(first: &'a str, second: &'a str) -> Self {
        let mut parts = Self::new();
        parts.push(first);
        parts.push(second);
        parts
    }

    fn push(&mut self, part: &'a str) {
        if let Some(parts) = &mut self.overflow {
            parts.push(part);
            return;
        }

        if self.len < MAX_INLINE_EXPLICIT_PARTS {
            self.inline[self.len] = part;
            self.len += 1;
            return;
        }

        let mut parts = Vec::with_capacity(MAX_INLINE_EXPLICIT_PARTS * 2);
        parts.extend_from_slice(&self.inline[..self.len]);
        parts.push(part);
        self.overflow = Some(parts);
    }

    fn len(&self) -> usize {
        self.overflow.as_ref().map_or(self.len, std::vec::Vec::len)
    }

    fn as_slice(&self) -> &[&'a str] {
        self.overflow.as_deref().unwrap_or(&self.inline[..self.len])
    }
}

fn explicit_hasant_chain_start_parts(unit: &PhoneticUnit) -> Option<BorrowedParts<'_>> {
    match unit.unit_type {
        PhoneticUnitType::Consonant => Some(BorrowedParts::from_one(unit.text.as_str())),
        PhoneticUnitType::RephOverConsonant => {
            let base = reph_base_part(unit)?;
            Some(BorrowedParts::from_two("rr", base))
        }
        PhoneticUnitType::SpecialForm if unit.text == "rr" => Some(BorrowedParts::from_one("rr")),
        _ => None,
    }
}

fn explicit_hasant_chain_next_part<'a>(
    parts: &[&str],
    unit: &'a PhoneticUnit,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> Option<&'a str> {
    match unit.unit_type {
        PhoneticUnitType::Consonant if is_explicit_phola_marker(&unit.text) => {
            if explicit_hasant_chain_with_next_part_is_valid(parts, &unit.text, conjunct_defs) {
                Some(unit.text.as_str())
            } else {
                None
            }
        }
        PhoneticUnitType::Consonant => Some(unit.text.as_str()),
        PhoneticUnitType::Unknown if unit.text == "w" => {
            if explicit_hasant_chain_with_next_part_is_valid(parts, "w", conjunct_defs) {
                Some("w")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn explicit_hasant_chain_with_next_part_is_valid(
    parts: &[&str],
    next: &str,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> bool {
    if parts.len() < MAX_INLINE_EXPLICIT_PARTS {
        let mut borrowed = [""; MAX_INLINE_EXPLICIT_PARTS];
        borrowed[..parts.len()].copy_from_slice(parts);
        borrowed[parts.len()] = next;
        return conjunct_defs.can_form_conjunct_from_parts(&borrowed[..parts.len() + 1]);
    }

    let mut borrowed = Vec::with_capacity(parts.len() + 1);
    borrowed.extend_from_slice(parts);
    borrowed.push(next);
    conjunct_defs.can_form_conjunct_from_parts(borrowed.as_slice())
}

fn explicit_hasant_chain_is_renderable(
    parts: &[&str],
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> bool {
    if !parts.iter().any(|part| is_explicit_phola_marker(part)) {
        return true;
    }

    conjunct_defs.can_form_conjunct_from_parts(parts)
}

fn is_explicit_phola_marker(part: &str) -> bool {
    matches!(part, "w" | "y" | "Y")
}

fn normalize_reph_and_vocalic_r(units: &mut Vec<PhoneticUnit>) {
    let mut read = 0;
    let mut write = 0;

    while read < units.len() {
        if read + 1 < units.len()
            && units[read].text == "rr"
            && units[read].unit_type == PhoneticUnitType::SpecialForm
            && units[read + 1].text == "i"
            && units[read + 1].unit_type == PhoneticUnitType::Vowel
        {
            let position = units[read].position;
            units[write] = PhoneticUnit {
                text: String::from("rri"),
                unit_type: PhoneticUnitType::Vowel,
                position,
            };
            read += 2;
            write += 1;
            continue;
        }

        if read + 1 < units.len()
            && units[read].text == "rr"
            && units[read].unit_type == PhoneticUnitType::SpecialForm
            && units[read + 1].unit_type == PhoneticUnitType::Consonant
        {
            let position = units[read].position;
            let next_text = units[read + 1].text.as_str();
            let mut reph_text = String::with_capacity(2 + next_text.len());
            reph_text.push_str("rr");
            reph_text.push_str(next_text);

            units[write] = PhoneticUnit {
                text: reph_text,
                unit_type: PhoneticUnitType::RephOverConsonant,
                position,
            };
            read += 2;
            write += 1;
            continue;
        }

        move_unit(units, read, write);

        read += 1;
        write += 1;
    }

    units.truncate(write);
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

fn normalize_redundant_reph_hasant(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_redundant_reph_hasant(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_redundant_reph_hasant_at(units, read) {
            move_unit(units, read, write);
            read += 2;
            write += 1;
            continue;
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

fn first_redundant_reph_hasant(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_redundant_reph_hasant_at(units, index))
}

fn is_redundant_reph_hasant_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && units[index].unit_type == PhoneticUnitType::SpecialForm
        && units[index].text == "rr"
        && units[index + 1].unit_type == PhoneticUnitType::ConsonantWithHasant
        && is_reph_target_after_redundant_hasant(&units[index + 2])
}

fn is_reph_target_after_redundant_hasant(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant || is_khanda_ta_unit(unit)
}

fn normalize_redundant_khanda_ta_hasant(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_redundant_khanda_ta_hasant(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_redundant_khanda_ta_hasant_at(units, read) {
            move_unit(units, read, write);
            read += 2;
            write += 1;
            continue;
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

fn first_redundant_khanda_ta_hasant(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_redundant_khanda_ta_hasant_at(units, index))
}

fn is_redundant_khanda_ta_hasant_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && is_khanda_ta_unit(&units[index])
        && units[index + 1].unit_type == PhoneticUnitType::ConsonantWithHasant
        && units[index + 2].unit_type == PhoneticUnitType::Consonant
}

fn is_khanda_ta_unit(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::SpecialForm && matches!(unit.text.as_str(), "t``" | "T``")
}

fn longest_conjunct_prefix_in_range(
    units: &[PhoneticUnit],
    start: usize,
    end: usize,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> Option<usize> {
    let mut node = conjunct_defs.conjunct_match_root();
    let mut best_length = None;

    'candidate: for (current, unit) in units.iter().enumerate().take(end).skip(start) {
        if unit.text.is_empty() {
            break;
        }

        for part in unit.text.split(",,") {
            let Some(next_node) = conjunct_defs.advance_conjunct_match(node, part) else {
                break 'candidate;
            };
            node = next_node;
        }

        let length = current - start + 1;
        if length >= 2 && conjunct_defs.conjunct_match_value(node).is_some() {
            best_length = Some(length);
        }
    }

    best_length.or_else(|| reph_tail_conjunct_prefix_in_range(units, start, end, conjunct_defs))
}

fn reph_tail_conjunct_prefix_in_range(
    units: &[PhoneticUnit],
    start: usize,
    end: usize,
    conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
) -> Option<usize> {
    let first = units.get(start)?;
    let reph_base = reph_base_part(first)?;
    let mut tail_parts = BorrowedParts::from_one(reph_base);
    let mut best_length = None;

    for (current, unit) in units.iter().enumerate().take(end).skip(start + 1) {
        if unit.text.is_empty() {
            break;
        }

        for part in unit.text.split(",,") {
            tail_parts.push(part);
        }

        let length = current - start + 1;
        if tail_parts.len() >= 2
            && conjunct_defs.can_form_conjunct_from_parts(tail_parts.as_slice())
            && !is_ambiguous_reph_r_phola_before_vowel(units, start, length, tail_parts.as_slice())
        {
            best_length = Some(length);
        }
    }

    best_length
}

fn is_ambiguous_reph_r_phola_before_vowel(
    units: &[PhoneticUnit],
    start: usize,
    length: usize,
    tail_parts: &[&str],
) -> bool {
    tail_parts.last() == Some(&"r")
        && units.get(start + length).is_some_and(|unit| {
            matches!(
                unit.unit_type,
                PhoneticUnitType::Vowel | PhoneticUnitType::TerminatingVowel
            )
        })
}

fn conjunct_text_for_range(units: &[PhoneticUnit], start: usize, length: usize) -> String {
    let mut conjunct_text = String::new();

    for unit in &units[start..start + length] {
        push_conjunct_text_parts(&mut conjunct_text, unit);
    }

    conjunct_text
}

fn push_conjunct_text_parts(conjunct_text: &mut String, unit: &PhoneticUnit) {
    if let Some(reph_base) = reph_base_part(unit) {
        push_conjunct_text_part(conjunct_text, "rr");
        push_conjunct_text_part(conjunct_text, reph_base);
        return;
    }

    for part in unit.text.split(",,") {
        push_conjunct_text_part(conjunct_text, part);
    }
}

fn push_conjunct_text_part(conjunct_text: &mut String, part: &str) {
    if !conjunct_text.is_empty() {
        conjunct_text.push_str(",,");
    }
    conjunct_text.push_str(part);
}

fn reph_base_part(unit: &PhoneticUnit) -> Option<&str> {
    if unit.unit_type == PhoneticUnitType::RephOverConsonant {
        unit.text.strip_prefix("rr").filter(|part| !part.is_empty())
    } else {
        None
    }
}

fn is_conjunct_run_component(unit: &PhoneticUnit) -> bool {
    matches!(
        unit.unit_type,
        PhoneticUnitType::Consonant
            | PhoneticUnitType::Conjunct
            | PhoneticUnitType::RephOverConsonant
    ) || (unit.unit_type == PhoneticUnitType::SpecialForm && unit.text == "rr")
        || (unit.unit_type == PhoneticUnitType::Unknown && unit.text == "w")
}

fn normalize_velar_nasal_conjunct_aliases(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_velar_nasal_conjunct_alias(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_velar_nasal_conjunct_alias_at(units, read) {
            if let Some(canonical_tail) = velar_nasal_conjunct_tail(&units[read + 1].text) {
                let position = units[read].position;
                let mut text = String::with_capacity(4 + canonical_tail.len());
                text.push_str("Ng,,");
                text.push_str(canonical_tail);

                units[write] = PhoneticUnit {
                    text,
                    unit_type: PhoneticUnitType::Conjunct,
                    position,
                };
                read += 2;
                write += 1;
                continue;
            }
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

fn first_velar_nasal_conjunct_alias(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(1)).find(|&index| {
        is_velar_nasal_conjunct_alias_at(units, index)
            && velar_nasal_conjunct_tail(&units[index + 1].text).is_some()
    })
}

fn is_velar_nasal_conjunct_alias_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 1 < units.len()
        && units[index].unit_type == PhoneticUnitType::SpecialForm
        && units[index].text == "ng"
        && units[index + 1].unit_type == PhoneticUnitType::Consonant
}

fn velar_nasal_conjunct_tail(text: &str) -> Option<&'static str> {
    match text {
        "g" => Some("g"),
        "gh" | "Gh" | "GH" => Some("gh"),
        _ => None,
    }
}

struct TrailingDiacritics<'a> {
    text: &'a str,
    offset: usize,
}

impl TrailingDiacritics<'_> {
    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

fn split_trailing_diacritics(word: &str) -> (&str, TrailingDiacritics<'_>) {
    let mut base_end = word.len();

    for (position, marker) in word.char_indices().rev() {
        if !matches!(marker, '^' | ':') {
            break;
        }

        base_end = position;
    }

    (
        &word[..base_end],
        TrailingDiacritics {
            text: &word[base_end..],
            offset: base_end,
        },
    )
}

fn append_trailing_diacritics(units: &mut Vec<PhoneticUnit>, suffix: TrailingDiacritics<'_>) {
    units.extend(
        suffix
            .text
            .char_indices()
            .map(|(offset, marker)| PhoneticUnit {
                text: marker.to_string(),
                unit_type: PhoneticUnitType::SpecialForm,
                position: suffix.offset + offset,
            }),
    );
}

fn normalize_iyw_long_iya_signal(units: &mut Vec<PhoneticUnit>) -> bool {
    let Some(first_match) = first_iyw_long_iya_signal(units) else {
        return false;
    };

    let mut read = first_match;
    let mut write = first_match;
    while read < units.len() {
        if is_iyw_long_iya_signal_at(units, read) {
            promote_short_i_to_long_i(&mut units[read]);
            move_unit(units, read, write);
            move_unit(units, read + 1, write + 1);
            write += 2;
            read += 3;
            continue;
        }

        move_unit(units, read, write);
        write += 1;
        read += 1;
    }

    units.truncate(write);
    true
}

fn first_iyw_long_iya_signal(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_iyw_long_iya_signal_at(units, index))
}

fn is_iyw_long_iya_signal_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && is_ya_consonant_unit(&units[index + 1])
        && is_long_iya_marker(&units[index + 2])
        && is_short_i_vowel_bearing_unit(&units[index])
}

fn is_short_i_vowel_bearing_unit(unit: &PhoneticUnit) -> bool {
    if !matches!(
        unit.unit_type,
        PhoneticUnitType::ConsonantWithVowel
            | PhoneticUnitType::ConjunctWithVowel
            | PhoneticUnitType::RephOverConsonantWithVowel
    ) {
        return false;
    }

    attached_vowel_key(&unit.text) == Some("i")
}

fn attached_vowel_key(text: &str) -> Option<&str> {
    let component = text.rsplit(",,").next()?;
    let component = component
        .strip_prefix("rr")
        .filter(|component| !component.is_empty())
        .unwrap_or(component);

    split_component_vowel_key(component)
}

fn split_component_vowel_key(component: &str) -> Option<&str> {
    for (boundary, _) in component.char_indices().skip(1) {
        let consonant = &component[..boundary];
        let vowel = &component[boundary..];

        if vowel_value(vowel).is_some()
            && (consonant_value(consonant).is_some() || is_phola_component_consonant(consonant))
        {
            return Some(vowel);
        }
    }

    None
}

fn is_phola_component_consonant(component: &str) -> bool {
    component == "w"
}

fn is_ya_consonant_unit(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant && matches!(unit.text.as_str(), "y" | "Y")
}

fn is_long_iya_marker(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Unknown && unit.text == "w"
}

fn promote_short_i_to_long_i(unit: &mut PhoneticUnit) {
    debug_assert!(is_short_i_vowel_bearing_unit(unit));
    unit.text.pop();
    unit.text.push('I');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(text: &str, unit_type: PhoneticUnitType, position: usize) -> PhoneticUnit {
        PhoneticUnit {
            text: text.to_string(),
            unit_type,
            position,
        }
    }

    #[test]
    fn compact_units_and_attach_vowels_skips_empty_units_in_place() {
        let mut units = vec![
            unit("k", PhoneticUnitType::Consonant, 0),
            unit("", PhoneticUnitType::Consonant, 1),
            unit("A", PhoneticUnitType::Vowel, 2),
            unit("", PhoneticUnitType::Consonant, 3),
            unit("rrk", PhoneticUnitType::RephOverConsonant, 4),
            unit("o", PhoneticUnitType::TerminatingVowel, 7),
            unit("ng", PhoneticUnitType::SpecialForm, 8),
        ];
        let capacity = units.capacity();

        compact_units_and_attach_vowels(&mut units);

        assert_eq!(units.capacity(), capacity);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type, unit.position))
                .collect::<Vec<_>>(),
            vec![
                ("kA", PhoneticUnitType::ConsonantWithVowel, 0),
                ("rrko", PhoneticUnitType::RephOverConsonantWithTerminator, 4),
                ("ng", PhoneticUnitType::SpecialForm, 8),
            ]
        );
    }
}
