use crate::definitions::{consonant_value, vowel_value};

#[inline]
pub(super) fn split_reph_consonant_vowel(text: &str) -> Option<(&str, &str)> {
    split_consonant_vowel(text.strip_prefix("rr")?)
}

#[inline]
pub(super) fn split_conjunct_component_vowel(text: &str) -> Option<(&str, &str)> {
    split_consonant_vowel(text).or_else(|| split_phola_component_vowel(text))
}

#[inline]
fn split_phola_component_vowel(text: &str) -> Option<(&str, &str)> {
    let vowel = text.strip_prefix('w')?;
    if !vowel.is_empty() && vowel_value(vowel).is_some() {
        Some(("w", vowel))
    } else {
        None
    }
}

#[inline]
pub(super) fn split_consonant_vowel(text: &str) -> Option<(&str, &str)> {
    for (boundary, _) in text.char_indices().skip(1) {
        let consonant = &text[..boundary];
        let vowel = &text[boundary..];

        if consonant_value(consonant).is_some() && vowel_value(vowel).is_some() {
            return Some((consonant, vowel));
        }
    }

    None
}
