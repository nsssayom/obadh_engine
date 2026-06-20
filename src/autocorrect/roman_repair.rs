use crate::{PhoneticUnit, PhoneticUnitType, Tokenizer};

pub const DEFAULT_ROMAN_REPAIR_BEAM_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct RomanRepairOptions {
    pub max_repairs: usize,
}

impl Default for RomanRepairOptions {
    fn default() -> Self {
        Self {
            max_repairs: DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomanRepair {
    pub text: String,
    pub cost: u16,
    pub kind: RomanRepairKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomanRepairedOutput {
    pub roman_input: String,
    pub bangla_output: String,
    pub repair_kind: &'static str,
    pub repair_cost: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RomanRepairKind {
    Original,
    InsertedSeparatorO,
}

impl RomanRepairKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::InsertedSeparatorO => "inserted_separator_o",
        }
    }
}

pub fn roman_repair_beam(input: &str, options: RomanRepairOptions) -> Vec<RomanRepair> {
    if input.is_empty() || options.max_repairs == 0 {
        return Vec::new();
    }

    let tokenizer = Tokenizer::new();
    let units = tokenizer.tokenize_word(input);
    let mut repairs = Vec::with_capacity(options.max_repairs.min(input.len() + 1));

    repairs.push(RomanRepair {
        text: input.to_string(),
        cost: 0,
        kind: RomanRepairKind::Original,
    });

    for byte_index in missing_separator_positions(input, &units) {
        if repairs.len() >= options.max_repairs {
            break;
        }

        let mut repaired = String::with_capacity(input.len() + 1);
        repaired.push_str(&input[..byte_index]);
        repaired.push('o');
        repaired.push_str(&input[byte_index..]);

        repairs.push(RomanRepair {
            text: repaired,
            cost: 1,
            kind: RomanRepairKind::InsertedSeparatorO,
        });
    }

    repairs
}

pub fn roman_repaired_outputs<F>(
    input: &str,
    baseline_output: &str,
    options: RomanRepairOptions,
    mut transliterate: F,
) -> Vec<RomanRepairedOutput>
where
    F: FnMut(&str) -> String,
{
    roman_repair_beam(input, options)
        .into_iter()
        .filter(|repair| repair.cost > 0)
        .filter_map(|repair| {
            let output = transliterate(&repair.text);
            (output != baseline_output).then_some(RomanRepairedOutput {
                roman_input: repair.text,
                bangla_output: output,
                repair_kind: repair.kind.as_str(),
                repair_cost: repair.cost,
            })
        })
        .collect()
}

fn missing_separator_positions(input: &str, units: &[PhoneticUnit]) -> Vec<usize> {
    let mut positions = Vec::new();

    for unit in units {
        if !is_conjunct_like(unit.unit_type) {
            continue;
        }

        append_conjunct_separator_positions(input, unit, &mut positions);
    }

    positions.sort_unstable();
    positions.dedup();
    positions
}

fn append_conjunct_separator_positions(
    input: &str,
    unit: &PhoneticUnit,
    positions: &mut Vec<usize>,
) {
    let mut parts = unit.text.split(",,");
    let Some(mut left) = parts.next() else {
        return;
    };

    let mut offset = 0_usize;
    for right in parts {
        offset += left.len();

        if !same_roman_component(left, right) {
            let byte_index = unit.position + offset;
            if is_insertable_boundary(input, byte_index) {
                positions.push(byte_index);
            }
        }

        left = right;
    }
}

fn is_conjunct_like(unit_type: PhoneticUnitType) -> bool {
    matches!(
        unit_type,
        PhoneticUnitType::Conjunct
            | PhoneticUnitType::ConjunctWithVowel
            | PhoneticUnitType::ConjunctWithTerminator
            | PhoneticUnitType::RephOverConsonant
            | PhoneticUnitType::RephOverConsonantWithVowel
            | PhoneticUnitType::RephOverConsonantWithTerminator
    )
}

fn same_roman_component(left: &str, right: &str) -> bool {
    roman_component_base(left).eq_ignore_ascii_case(roman_component_base(right))
}

fn roman_component_base(component: &str) -> &str {
    component
        .strip_suffix('a')
        .or_else(|| component.strip_suffix('A'))
        .or_else(|| component.strip_suffix('i'))
        .or_else(|| component.strip_suffix('I'))
        .or_else(|| component.strip_suffix('u'))
        .or_else(|| component.strip_suffix('U'))
        .or_else(|| component.strip_suffix('e'))
        .or_else(|| component.strip_suffix('E'))
        .or_else(|| component.strip_suffix('o'))
        .or_else(|| component.strip_suffix('O'))
        .unwrap_or(component)
}

fn is_insertable_boundary(input: &str, byte_index: usize) -> bool {
    byte_index > 0
        && input.is_char_boundary(byte_index)
        && input
            .as_bytes()
            .get(byte_index)
            .is_some_and(u8::is_ascii_alphabetic)
        && input
            .as_bytes()
            .get(byte_index - 1)
            .is_some_and(u8::is_ascii_alphabetic)
}

#[cfg(test)]
mod tests {
    use super::{roman_repair_beam, RomanRepairKind, RomanRepairOptions};

    #[test]
    fn inserts_separator_o_at_implicit_conjunct_boundary() {
        let repairs = roman_repair_beam("okalpokk", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "okalopokk"
                && repair.cost == 1
                && repair.kind == RomanRepairKind::InsertedSeparatorO
        }));
    }

    #[test]
    fn does_not_split_repeated_consonants() {
        let repairs = roman_repair_beam("biggan", RomanRepairOptions::default());

        assert!(!repairs.iter().any(|repair| repair.text == "bigogan"));
    }

    #[test]
    fn returns_only_original_for_zero_sized_beam() {
        let repairs = roman_repair_beam("okalpokk", RomanRepairOptions { max_repairs: 0 });

        assert!(repairs.is_empty());
    }
}
