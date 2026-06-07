use crate::definitions::conjuncts::conjuncts;

use super::super::tokenizer::{PhoneticUnit, PhoneticUnitType, Token, TokenType};

#[derive(Clone, Copy, Default)]
pub(super) struct TokenNumberBoundary {
    can_contain_number: bool,
    ends_with_number: bool,
}

impl TokenNumberBoundary {
    #[inline]
    pub(super) fn from_word(word: &str) -> Self {
        Self {
            can_contain_number: true,
            ends_with_number: text_ends_with_number(word),
        }
    }
}

#[inline]
pub(super) fn is_decimal_separator_at(
    text: &str,
    byte_index: usize,
    current_char_len: usize,
    current_word: Option<&str>,
    previous_boundary: TokenNumberBoundary,
) -> bool {
    if &text[byte_index..byte_index + current_char_len] != "." {
        return false;
    }

    let previous = current_word.map_or(previous_boundary, TokenNumberBoundary::from_word);

    previous.can_contain_number
        && previous.ends_with_number
        && next_token_starts_with_number(text, byte_index + current_char_len)
}

#[inline]
fn next_token_starts_with_number(text: &str, byte_index: usize) -> bool {
    text.get(byte_index..)
        .and_then(|suffix| suffix.chars().next())
        .is_some_and(|character| character.is_numeric())
}

#[inline]
fn text_ends_with_number(text: &str) -> bool {
    text.chars()
        .next_back()
        .is_some_and(|character| character.is_numeric())
}

#[inline]
fn is_cluster_unit(unit: &PhoneticUnit) -> bool {
    matches!(
        unit.unit_type,
        PhoneticUnitType::Conjunct
            | PhoneticUnitType::ConjunctWithVowel
            | PhoneticUnitType::ConjunctWithTerminator
            | PhoneticUnitType::RephOverConsonant
            | PhoneticUnitType::RephOverConsonantWithVowel
            | PhoneticUnitType::RephOverConsonantWithTerminator
    )
}

#[inline]
pub(super) fn starts_with_cluster(units: &[PhoneticUnit]) -> bool {
    if units.first().is_some_and(is_cluster_unit) {
        return true;
    }

    let [first, second, ..] = units else {
        return false;
    };

    if first.unit_type != PhoneticUnitType::Consonant
        || !matches!(
            second.unit_type,
            PhoneticUnitType::Consonant | PhoneticUnitType::Unknown
        )
        || !matches!(second.text.as_str(), "y" | "Y" | "w")
    {
        return false;
    }

    conjuncts().can_form_conjunct_from_parts(&[first.text.as_str(), second.text.as_str()])
}

#[inline]
pub(super) fn is_decimal_separator(tokens: &[Token], index: usize) -> bool {
    if tokens[index].content != "." || index == 0 || index + 1 >= tokens.len() {
        return false;
    }

    let previous = &tokens[index - 1];
    let next = &tokens[index + 1];

    if previous.token_type == TokenType::Number && next.token_type == TokenType::Number {
        return true;
    }

    token_can_contain_number(previous)
        && token_can_contain_number(next)
        && token_ends_with_number(previous)
        && token_starts_with_number(next)
}

#[inline]
fn token_can_contain_number(token: &Token) -> bool {
    matches!(token.token_type, TokenType::Word | TokenType::Number)
}

#[inline]
fn token_ends_with_number(token: &Token) -> bool {
    token
        .content
        .chars()
        .next_back()
        .is_some_and(|character| character.is_numeric())
}

#[inline]
fn token_starts_with_number(token: &Token) -> bool {
    token
        .content
        .chars()
        .next()
        .is_some_and(|character| character.is_numeric())
}
