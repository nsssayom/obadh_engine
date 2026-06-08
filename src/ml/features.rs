use serde::{Deserialize, Serialize};

use crate::engine::{PhoneticUnit, PhoneticUnitType, Token, TokenType, Transliterator};

/// Versioned schema name for the first Obadh ML feature contract.
pub const FEATURE_SCHEMA_VERSION: &str = "obadh.ml.features.v0";

/// Every phonetic unit is expanded into three CTC time slots.
pub const FEATURE_SLOTS_PER_UNIT: usize = 3;

const SLOT_BEFORE: &str = "before";
const SLOT_MAIN: &str = "main";
const SLOT_AFTER: &str = "after";

/// A deterministic feature document for one input string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlFeatureDocument {
    pub schema: String,
    pub engine_version: String,
    pub input: String,
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sanitization_error: Option<String>,
    pub deterministic: String,
    pub tokens: Vec<MlTokenFeatures>,
}

/// Feature metadata for one tokenizer token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlTokenFeatures {
    pub index: usize,
    pub content: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub token_type: String,
    pub deterministic: Option<String>,
    pub units: Vec<MlPhoneticUnitFeatures>,
    pub slots: Vec<MlFeatureSlot>,
}

/// Feature metadata for one Obadh phonetic unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlPhoneticUnitFeatures {
    pub index: usize,
    pub roman: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub unit_type: String,
    pub rule_id: String,
}

/// One integer-vocabulary-ready CTC feature slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlFeatureSlot {
    pub index: usize,
    pub unit_index: usize,
    pub slot_type: String,
    pub roman: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub unit_type: String,
    pub rule_id: String,
    pub feature_key: String,
}

/// Extract a versioned ML feature document for one input string.
pub fn extract_features(transliterator: &Transliterator, input: &str) -> MlFeatureDocument {
    match transliterator.sanitize(input) {
        Ok(sanitized) => extract_accepted_features(transliterator, input, &sanitized),
        Err(error) => MlFeatureDocument {
            schema: FEATURE_SCHEMA_VERSION.to_string(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            input: input.to_string(),
            accepted: false,
            sanitization_error: Some(error),
            deterministic: transliterator.transliterate(input),
            tokens: Vec::new(),
        },
    }
}

fn extract_accepted_features(
    transliterator: &Transliterator,
    input: &str,
    sanitized: &str,
) -> MlFeatureDocument {
    let tokens = transliterator.tokenize(sanitized);
    let deterministic = transliterator.transliterate_tokens(&tokens);
    let token_features = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| extract_token_features(transliterator, &tokens, index, token))
        .collect();

    MlFeatureDocument {
        schema: FEATURE_SCHEMA_VERSION.to_string(),
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        input: input.to_string(),
        accepted: true,
        sanitization_error: None,
        deterministic,
        tokens: token_features,
    }
}

fn extract_token_features(
    transliterator: &Transliterator,
    tokens: &[Token],
    index: usize,
    token: &Token,
) -> MlTokenFeatures {
    let deterministic = transliterator.transliterate_token_at(tokens, index);
    let (units, slots) = if token.token_type == TokenType::Word {
        let units = transliterator.tokenize_phonetic(&token.content);
        let unit_features = extract_unit_features(&units);
        let slots = expand_slots(&unit_features);
        (unit_features, slots)
    } else {
        (Vec::new(), Vec::new())
    };

    MlTokenFeatures {
        index,
        content: token.content.clone(),
        byte_start: token.position,
        byte_end: token.position + token.content.len(),
        token_type: token_type_name(&token.token_type).to_string(),
        deterministic,
        units,
        slots,
    }
}

fn extract_unit_features(units: &[PhoneticUnit]) -> Vec<MlPhoneticUnitFeatures> {
    units
        .iter()
        .enumerate()
        .map(|(index, unit)| {
            let unit_type = unit_type_name(unit.unit_type);
            MlPhoneticUnitFeatures {
                index,
                roman: unit.text.clone(),
                byte_start: unit.position,
                byte_end: unit.position + unit.text.len(),
                unit_type: unit_type.to_string(),
                rule_id: rule_id(unit_type, &unit.text),
            }
        })
        .collect()
}

fn expand_slots(units: &[MlPhoneticUnitFeatures]) -> Vec<MlFeatureSlot> {
    let mut slots = Vec::with_capacity(units.len() * FEATURE_SLOTS_PER_UNIT);

    for unit in units {
        for slot_type in [SLOT_BEFORE, SLOT_MAIN, SLOT_AFTER] {
            let feature_key = feature_key(slot_type, &unit.rule_id);
            slots.push(MlFeatureSlot {
                index: slots.len(),
                unit_index: unit.index,
                slot_type: slot_type.to_string(),
                roman: unit.roman.clone(),
                byte_start: unit.byte_start,
                byte_end: unit.byte_end,
                unit_type: unit.unit_type.clone(),
                rule_id: unit.rule_id.clone(),
                feature_key,
            });
        }
    }

    slots
}

fn token_type_name(token_type: &TokenType) -> &'static str {
    match token_type {
        TokenType::Word => "word",
        TokenType::Punctuation => "punctuation",
        TokenType::Whitespace => "whitespace",
        TokenType::Number => "number",
        TokenType::Symbol => "symbol",
    }
}

fn unit_type_name(unit_type: PhoneticUnitType) -> &'static str {
    match unit_type {
        PhoneticUnitType::Consonant => "consonant",
        PhoneticUnitType::Vowel => "vowel",
        PhoneticUnitType::TerminatingVowel => "terminating_vowel",
        PhoneticUnitType::ConsonantWithVowel => "consonant_with_vowel",
        PhoneticUnitType::ConsonantWithTerminator => "consonant_with_terminator",
        PhoneticUnitType::ConsonantWithHasant => "consonant_with_hasant",
        PhoneticUnitType::Conjunct => "conjunct",
        PhoneticUnitType::ConjunctWithVowel => "conjunct_with_vowel",
        PhoneticUnitType::ConjunctWithTerminator => "conjunct_with_terminator",
        PhoneticUnitType::RephOverConsonant => "reph_over_consonant",
        PhoneticUnitType::RephOverConsonantWithVowel => "reph_over_consonant_with_vowel",
        PhoneticUnitType::RephOverConsonantWithTerminator => "reph_over_consonant_with_terminator",
        PhoneticUnitType::SpecialForm => "special_form",
        PhoneticUnitType::Numeral => "numeral",
        PhoneticUnitType::Symbol => "symbol",
        PhoneticUnitType::Unknown => "unknown",
    }
}

fn rule_id(unit_type: &str, roman: &str) -> String {
    format!("{unit_type}:{roman}")
}

fn feature_key(slot_type: &str, rule_id: &str) -> String {
    format!("{slot_type}|{rule_id}")
}
