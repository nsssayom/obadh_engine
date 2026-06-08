use super::{Token, TokenType};
use crate::engine::text_boundary::{
    is_explicit_hasant_signal_at, is_khanda_ta_suffix_signal_at, is_phonetic_mark_signal,
};

pub(super) fn tokenize_text(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_word = CurrentWord::new();

    let mut i = 0;
    while i < text.len() {
        let c = text[i..].chars().next().unwrap();
        let char_len = c.len_utf8();

        if is_phonetic_mark_signal(c) {
            current_word.push_signal_char(c, i);
            i += char_len;
            continue;
        }

        if !current_word.is_empty()
            && is_khanda_ta_suffix_signal_at(c, text, i, current_word.as_str())
        {
            current_word.push_signal("``", i);
            i += 2;
            continue;
        }

        if is_explicit_hasant_signal_at(c, text, i) {
            current_word.push_signal(",,", i);
            i += 2;
            continue;
        }

        if c.is_whitespace() {
            current_word.flush_into(&mut tokens);
            tokens.push(Token {
                content: c.to_string(),
                token_type: TokenType::Whitespace,
                position: i,
            });
        } else if c.is_ascii_punctuation() {
            current_word.flush_into(&mut tokens);
            tokens.push(Token {
                content: c.to_string(),
                token_type: TokenType::Punctuation,
                position: i,
            });
        } else if !c.is_alphanumeric() {
            current_word.flush_into(&mut tokens);
            tokens.push(Token {
                content: c.to_string(),
                token_type: TokenType::Symbol,
                position: i,
            });
        } else {
            current_word.push_alphanumeric(c, i);
        }

        i += char_len;
    }

    current_word.flush_into(&mut tokens);

    tokens
}

struct CurrentWord {
    text: String,
    position: usize,
    is_number: bool,
}

impl CurrentWord {
    fn new() -> Self {
        Self {
            text: String::new(),
            position: 0,
            is_number: true,
        }
    }

    fn as_str(&self) -> &str {
        &self.text
    }

    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn push_signal_char(&mut self, signal: char, position: usize) {
        self.start_if_empty(position);
        self.is_number = false;
        self.text.push(signal);
    }

    fn push_signal(&mut self, signal: &str, position: usize) {
        self.start_if_empty(position);
        self.is_number = false;
        self.text.push_str(signal);
    }

    fn push_alphanumeric(&mut self, character: char, position: usize) {
        self.start_if_empty(position);
        self.is_number &= character.is_numeric();
        self.text.push(character);
    }

    fn flush_into(&mut self, tokens: &mut Vec<Token>) {
        if self.text.is_empty() {
            self.is_number = true;
            return;
        }

        let token_type = if self.is_number {
            TokenType::Number
        } else {
            TokenType::Word
        };

        let capacity = self.text.capacity();
        let content = std::mem::replace(&mut self.text, String::with_capacity(capacity));
        self.is_number = true;

        tokens.push(Token {
            content,
            token_type,
            position: self.position,
        });
    }

    fn start_if_empty(&mut self, position: usize) {
        if self.text.is_empty() {
            self.position = position;
            self.is_number = true;
        }
    }
}
