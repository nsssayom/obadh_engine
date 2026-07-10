use crate::definitions::{
    consonant_value,
    diacritics::{ANUSVARA, CHANDRABINDU, HASANT, KHANDA_TA, VISARGA},
    symbol_value,
};

use super::{
    components::{
        split_conjunct_component_vowel, split_consonant_vowel, split_reph_consonant_vowel,
    },
    parts::ConjunctParts,
    Transliterator,
};
use crate::engine::tokenizer::{PhoneticUnit, PhoneticUnitType};

impl Transliterator {
    pub(super) fn transliterate_word_units_into(
        &self,
        result: &mut String,
        word: &str,
        phonetic_units: &mut Vec<PhoneticUnit>,
    ) {
        self.tokenizer.tokenize_word_into(word, phonetic_units);

        let mut previous_unit_accepts_dependent_vowel = false;
        // `result` may already hold earlier words; the hasant rule below only
        // looks at what this word has rendered so far.
        let word_start = result.len();

        let unit_count = phonetic_units.len();
        for (unit_index, unit) in phonetic_units.iter().enumerate() {
            let is_last_unit = unit_index + 1 == unit_count;

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
                        if hasant_can_attach(&result[word_start..]) {
                            result.push_str(HASANT);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::Conjunct => {
                    let parts = ConjunctParts::from_text(&unit.text);

                    if !self.append_conjunct_parts(result, parts.as_slice()) {
                        result.push_str(&unit.text);
                    }
                    // A conjunct ends on a consonant, so it still takes a matra.
                    previous_unit_accepts_dependent_vowel = true;
                }
                PhoneticUnitType::ConjunctWithVowel => {
                    let mut parts = ConjunctParts::from_text(&unit.text);

                    if parts.len() >= 2 {
                        let last_part = parts.last().expect("parts length checked");
                        if let Some((last_consonant, vowel_part)) =
                            split_conjunct_component_vowel(last_part)
                        {
                            parts.replace_last(last_consonant);

                            if self.append_conjunct_parts(result, parts.as_slice()) {
                                if !self.append_dependent_vowel(result, vowel_part) {
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
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::ConjunctWithTerminator => {
                    let mut parts = ConjunctParts::from_text(&unit.text);

                    if parts.len() >= 2 {
                        let last_part = parts.last().expect("parts length checked");
                        if let Some((last_consonant, terminator_part)) =
                            split_conjunct_component_vowel(last_part)
                        {
                            parts.replace_last(last_consonant);

                            if self.append_conjunct_parts(result, parts.as_slice()) {
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
                        } else if !self.append_conjunct_parts(result, parts.as_slice()) {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::RephOverConsonant => {
                    if let Some(mapped) = self.conjuncts.create_conjunct(&unit.text) {
                        result.push_str(mapped);
                    } else {
                        let consonant_text = &unit.text[2..];

                        if let Some(bengali_consonant) = consonant_value(consonant_text) {
                            Self::append_reph_prefix(result);
                            result.push_str(bengali_consonant);
                        } else {
                            result.push_str(&unit.text);
                        }
                    }
                    // Reph sits over a consonant, which still takes a matra.
                    previous_unit_accepts_dependent_vowel = true;
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
                            Self::append_reph_prefix(result);
                            result.push_str(bengali_consonant);
                            if !self.append_dependent_vowel(result, vowel_part) {
                                result.push_str(vowel_part);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
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
                            Self::append_reph_prefix(result);
                            result.push_str(bengali_consonant);

                            if !terminator_part.is_empty()
                                && terminator_part != "o"
                                && !self.append_dependent_vowel(result, terminator_part)
                            {
                                result.push_str(terminator_part);
                            }
                        } else {
                            result.push_str(&unit.text);
                        }
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::SpecialForm => {
                    if unit.text == "rr" {
                        Self::append_reph_prefix(result);
                    } else if unit.text == "^" {
                        result.push_str(CHANDRABINDU);
                    } else if unit.text == ":" {
                        result.push_str(VISARGA);
                    } else if matches!(unit.text.as_str(), "t``" | "T``") {
                        result.push_str(KHANDA_TA);
                    } else if matches!(unit.text.as_str(), "ng" | "M") {
                        result.push_str(ANUSVARA);
                    } else {
                        result.push_str(&unit.text);
                    }
                    previous_unit_accepts_dependent_vowel = false;
                }
                PhoneticUnitType::Numeral => {
                    self.render_number_token(result, &unit.text);
                    previous_unit_accepts_dependent_vowel = false;
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

/// A hasant suppresses a consonant's inherent vowel, so it can only attach to a
/// consonant.
///
/// `rendered_word` is what this word has emitted so far. When it is empty the
/// `,,` signal is the documented standalone marker and renders on its own.
/// Otherwise the hasant is dropped unless it has a consonant to sit on: Bangla
/// has no cluster that stacks a hasant on a kar, on a chandrabindu, anusvar,
/// bisarga or khanda ta, on a numeral, or on another hasant.
fn hasant_can_attach(rendered_word: &str) -> bool {
    match rendered_word.chars().next_back() {
        None => true,
        Some(character) => bears_hasant(character),
    }
}

/// Whether a rendered Bengali character can carry a hasant: ক..হ, the phota that
/// completes ড়/ঢ়/য়, and those three in their precomposed forms.
///
/// Khanda ta (ৎ) is excluded — it is already a dead consonant.
fn bears_hasant(character: char) -> bool {
    matches!(
        character,
        '\u{0995}'..='\u{09B9}' | '\u{09BC}' | '\u{09DC}' | '\u{09DD}' | '\u{09DF}'
    )
}
