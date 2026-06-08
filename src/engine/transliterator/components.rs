use crate::definitions::{consonant_value, vowel_value, vowels::MAX_VOWEL_RULE_BYTES};

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
    let max_vowel_bytes = MAX_VOWEL_RULE_BYTES.min(text.len().saturating_sub(1));

    for vowel_len in (1..=max_vowel_bytes).rev() {
        let boundary = text.len() - vowel_len;
        let consonant = &text[..boundary];
        let vowel = &text[boundary..];

        if consonant_value(consonant).is_some() && vowel_value(vowel).is_some() {
            return Some((consonant, vowel));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_consonant_vowel_checks_longest_vowel_suffix_first() {
        assert_eq!(split_consonant_vowel("kaa"), Some(("k", "aa")));
        assert_eq!(split_consonant_vowel("kOU"), Some(("k", "OU")));
        assert_eq!(split_consonant_vowel("krri"), Some(("k", "rri")));
        assert_eq!(split_consonant_vowel("chhi"), Some(("chh", "i")));
    }

    #[test]
    fn split_reph_and_phola_vowel_units_share_bounded_split() {
        assert_eq!(split_reph_consonant_vowel("rrkrri"), Some(("k", "rri")));
        assert_eq!(split_conjunct_component_vowel("wA"), Some(("w", "A")));
    }

    #[test]
    fn split_consonant_vowel_rejects_missing_parts() {
        assert_eq!(split_consonant_vowel("k"), None);
        assert_eq!(split_consonant_vowel("aa"), None);
        assert_eq!(split_consonant_vowel("notarule"), None);
    }
}
