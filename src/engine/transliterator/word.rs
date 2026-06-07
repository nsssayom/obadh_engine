use std::borrow::Cow;

use crate::definitions::{consonant_value, diacritic_value, symbol_value, vowel_value};

use super::{
    boundary::starts_with_cluster,
    components::{
        split_conjunct_component_vowel, split_consonant_vowel, split_reph_consonant_vowel,
    },
    parts::ConjunctParts,
    Transliterator,
};
use crate::engine::tokenizer::{PhoneticUnit, PhoneticUnitType};

impl Transliterator {
    fn conjunct_component(&self, part: &str) -> Option<&'static str> {
        match part {
            "rZ" => Some("র\u{200C}"),
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

    fn append_vowel(&self, output: &mut String, vowel_key: &str, as_dependent: bool) -> bool {
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

    fn append_consonant_vowel(
        &self,
        output: &mut String,
        consonant_key: &str,
        vowel_key: &str,
    ) -> bool {
        let Some(bengali_consonant) = consonant_value(consonant_key) else {
            return false;
        };

        output.push_str(bengali_consonant);
        if !self.append_vowel(output, vowel_key, true) {
            output.push_str(vowel_key);
        }
        true
    }

    fn append_consonant_terminator(
        &self,
        output: &mut String,
        consonant_key: &str,
        terminator_key: &str,
    ) -> bool {
        let Some(bengali_consonant) = consonant_value(consonant_key) else {
            return false;
        };

        output.push_str(bengali_consonant);
        if terminator_key != "o" && !self.append_vowel(output, terminator_key, true) {
            output.push_str(terminator_key);
        }
        true
    }

    fn should_suppress_visible_a(&self, vowel_key: &str, following_units: &[PhoneticUnit]) -> bool {
        vowel_key == "a" && starts_with_cluster(following_units)
    }

    pub(super) fn transliterate_word_units_into(
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
                        result.push_str(&unit.text);
                        previous_unit_accepts_dependent_vowel = false;
                    }
                }
                PhoneticUnitType::Vowel => {
                    if self.append_vowel(
                        result,
                        unit.text.as_str(),
                        previous_unit_accepts_dependent_vowel,
                    ) {
                        previous_unit_accepts_dependent_vowel = false;
                    } else {
                        result.push_str(&unit.text);
                        previous_unit_accepts_dependent_vowel = false;
                    }
                }
                PhoneticUnitType::TerminatingVowel => {
                    if self.append_vowel(
                        result,
                        unit.text.as_str(),
                        previous_unit_accepts_dependent_vowel,
                    ) {
                        previous_unit_accepts_dependent_vowel = false;
                    } else {
                        result.push_str(&unit.text);
                        previous_unit_accepts_dependent_vowel = false;
                    }
                }
                PhoneticUnitType::ConsonantWithVowel => {
                    if let Some((consonant_part, vowel_part)) = split_consonant_vowel(&unit.text) {
                        if self.should_suppress_visible_a(vowel_part, following_units) {
                            if let Some(bengali_consonant) = consonant_value(consonant_part) {
                                result.push_str(bengali_consonant);
                                previous_unit_accepts_dependent_vowel = true;
                                continue;
                            }
                        }

                        if !self.append_consonant_vowel(result, consonant_part, vowel_part) {
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
                    if let Some((consonant_part, terminator_part)) =
                        split_consonant_vowel(&unit.text)
                    {
                        if !self.append_consonant_terminator(
                            result,
                            consonant_part,
                            terminator_part,
                        ) {
                            result.push_str(&unit.text);
                        }
                    } else if let Some(bengali_consonant) = consonant_value(unit.text.as_str()) {
                        result.push_str(bengali_consonant);
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::ConsonantWithHasant => {
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
                        let consonant_text = &unit.text[2..];

                        if let Some(bengali_consonant) = consonant_value(consonant_text) {
                            result.push_str("র্");
                            result.push_str(bengali_consonant);
                        } else {
                            result.push_str(&unit.text);
                        }
                    }
                }
                PhoneticUnitType::RephOverConsonantWithVowel => {
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
                        } else if let Some(bengali_consonant) = consonant_value(consonant_part) {
                            result.push_str("র্");
                            result.push_str(bengali_consonant);
                            if !self.append_vowel(result, vowel_part, true) {
                                result.push_str(vowel_part);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::RephOverConsonantWithTerminator => {
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
                            result.push_str("র্");
                            result.push_str(bengali_consonant);

                            if !terminator_part.is_empty()
                                && terminator_part != "o"
                                && !self.append_vowel(result, terminator_part, true)
                            {
                                result.push_str(terminator_part);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                }
                PhoneticUnitType::SpecialForm => {
                    if unit.text == "rr" {
                        result.push_str("র্");
                    } else if unit.text == "^" {
                        if let Some(chandrabindu) = diacritic_value("^") {
                            result.push_str(chandrabindu);
                        } else {
                            result.push('ঁ');
                        }
                    } else if unit.text == ":" {
                        if let Some(visarga) = diacritic_value(":") {
                            result.push_str(visarga);
                        } else {
                            result.push('ঃ');
                        }
                    } else if matches!(unit.text.as_str(), "t``" | "T``") {
                        let khanda_ta = diacritic_value(unit.text.as_str()).unwrap_or("ৎ");
                        result.push_str(khanda_ta);
                    } else if matches!(unit.text.as_str(), "ng" | "M") {
                        if let Some(anusvara) = diacritic_value(unit.text.as_str()) {
                            result.push_str(anusvara);
                        } else {
                            result.push('ং');
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::Numeral => {
                    self.render_number_token(result, &unit.text);
                }
                PhoneticUnitType::Symbol => {
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
