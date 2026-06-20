use super::super::tokenizer::{Token, TokenType};

#[derive(Clone, Copy, Default)]
pub(super) struct TokenNumberBoundary {
    can_contain_number: bool,
    ends_with_number: bool,
}

impl TokenNumberBoundary {
    #[inline]
    pub(super) const fn from_number_state(ends_with_number: bool) -> Self {
        Self {
            can_contain_number: true,
            ends_with_number,
        }
    }
}

#[inline]
pub(super) fn is_decimal_separator_at(
    text: &str,
    byte_index: usize,
    current_char_len: usize,
    current_word: Option<TokenNumberBoundary>,
    previous_boundary: TokenNumberBoundary,
) -> bool {
    if &text[byte_index..byte_index + current_char_len] != "." {
        return false;
    }

    let previous = current_word.unwrap_or(previous_boundary);

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
