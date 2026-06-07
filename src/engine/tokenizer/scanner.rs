use super::{Token, TokenType};

pub(super) fn tokenize_text(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_word = String::new();
    let mut current_position = 0;

    let mut i = 0;
    while i < text.len() {
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

        if !current_word.is_empty()
            && c == '`'
            && next_char(text, i, char_len) == Some('`')
            && ends_with_khanda_ta_base_signal(&current_word)
        {
            current_word.push_str("``");
            i += 2;
            continue;
        }

        if c == ',' && next_char(text, i, char_len) == Some(',') {
            if current_word.is_empty() {
                current_position = i;
            }

            current_word.push_str(",,");
            i += 2;
            continue;
        }

        if c.is_whitespace() {
            push_current_word_token(&mut current_word, current_position, &mut tokens);
            tokens.push(Token {
                content: c.to_string(),
                token_type: TokenType::Whitespace,
                position: i,
            });
            current_position = i + char_len;
        } else if c.is_ascii_punctuation() {
            push_current_word_token(&mut current_word, current_position, &mut tokens);
            tokens.push(Token {
                content: c.to_string(),
                token_type: TokenType::Punctuation,
                position: i,
            });
            current_position = i + char_len;
        } else if !c.is_alphanumeric() {
            push_current_word_token(&mut current_word, current_position, &mut tokens);
            tokens.push(Token {
                content: c.to_string(),
                token_type: TokenType::Symbol,
                position: i,
            });
            current_position = i + char_len;
        } else {
            if current_word.is_empty() {
                current_position = i;
            }
            current_word.push(c);
        }

        i += char_len;
    }

    push_current_word_token(&mut current_word, current_position, &mut tokens);

    tokens
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
