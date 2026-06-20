use super::Transliterator;
use crate::definitions::{consonant_value, vowel_value};

impl Transliterator {
    pub(super) fn append_dependent_vowel(&self, output: &mut String, vowel_key: &str) -> bool {
        let Some(vowel) = vowel_value(vowel_key) else {
            return false;
        };

        if let Some(dependent) = &vowel.dependent {
            output.push_str(dependent);
        }
        true
    }

    pub(super) fn append_vowel(
        &self,
        output: &mut String,
        vowel_key: &str,
        as_dependent: bool,
    ) -> bool {
        let Some(vowel) = vowel_value(vowel_key) else {
            return false;
        };

        if as_dependent {
            if let Some(dependent) = &vowel.dependent {
                output.push_str(dependent);
            } else {
                output.push_str(vowel.independent);
            }
        } else {
            output.push_str(vowel.independent);
        }

        true
    }

    pub(super) fn append_consonant_vowel(
        &self,
        output: &mut String,
        consonant_key: &str,
        vowel_key: &str,
    ) -> bool {
        let Some(bengali_consonant) = consonant_value(consonant_key) else {
            return false;
        };

        output.push_str(bengali_consonant);
        if !self.append_dependent_vowel(output, vowel_key) {
            output.push_str(vowel_key);
        }
        true
    }

    pub(super) fn append_consonant_terminator(
        &self,
        output: &mut String,
        consonant_key: &str,
        terminator_key: &str,
    ) -> bool {
        let Some(bengali_consonant) = consonant_value(consonant_key) else {
            return false;
        };

        output.push_str(bengali_consonant);
        if terminator_key != "o" && !self.append_dependent_vowel(output, terminator_key) {
            output.push_str(terminator_key);
        }
        true
    }
}
