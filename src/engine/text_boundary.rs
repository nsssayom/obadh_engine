//! Shared byte-scanner helpers for text token boundaries.

#[inline(always)]
pub(crate) fn is_phonetic_mark_signal(character: char) -> bool {
    matches!(character, '^' | ':')
}

#[inline(always)]
pub(crate) fn is_explicit_hasant_signal_at(character: char, text: &str, byte_index: usize) -> bool {
    character == ',' && text.as_bytes().get(byte_index + 1) == Some(&b',')
}

#[inline(always)]
pub(crate) fn is_khanda_ta_suffix_signal_at(
    character: char,
    text: &str,
    byte_index: usize,
    preceding_word: &str,
) -> bool {
    character == '`'
        && text.as_bytes().get(byte_index + 1) == Some(&b'`')
        && ends_with_khanda_ta_base_signal(preceding_word)
}

#[inline(always)]
fn ends_with_khanda_ta_base_signal(text: &str) -> bool {
    text.chars()
        .next_back()
        .is_some_and(|character| matches!(character, 't' | 'T'))
}
