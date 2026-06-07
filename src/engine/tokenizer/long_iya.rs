use crate::definitions::{consonant_value, vowel_value};

use super::{PhoneticUnit, PhoneticUnitType};

pub(super) fn is_long_iya_marker_at(text: &str, byte_index: usize) -> bool {
    byte_index > 0 && matches!(text.as_bytes().get(byte_index - 1), Some(b'y') | Some(b'Y'))
}

pub(super) fn normalize_iyw_long_iya_signal(units: &mut Vec<PhoneticUnit>) -> bool {
    let Some(first_match) = first_iyw_long_iya_signal(units) else {
        return false;
    };

    let mut read = first_match;
    let mut write = first_match;
    while read < units.len() {
        if is_iyw_long_iya_signal_at(units, read) {
            promote_short_i_to_long_i(&mut units[read]);
            super::move_unit(units, read, write);
            super::move_unit(units, read + 1, write + 1);
            write += 2;
            read += 3;
            continue;
        }

        super::move_unit(units, read, write);
        write += 1;
        read += 1;
    }

    units.truncate(write);
    true
}

fn first_iyw_long_iya_signal(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_iyw_long_iya_signal_at(units, index))
}

fn is_iyw_long_iya_signal_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && is_ya_consonant_unit(&units[index + 1])
        && is_long_iya_marker(&units[index + 2])
        && is_short_i_vowel_bearing_unit(&units[index])
}

fn is_short_i_vowel_bearing_unit(unit: &PhoneticUnit) -> bool {
    if !matches!(
        unit.unit_type,
        PhoneticUnitType::ConsonantWithVowel
            | PhoneticUnitType::ConjunctWithVowel
            | PhoneticUnitType::RephOverConsonantWithVowel
    ) {
        return false;
    }

    attached_vowel_key(&unit.text) == Some("i")
}

fn attached_vowel_key(text: &str) -> Option<&str> {
    let component = text.rsplit(",,").next()?;
    let component = component
        .strip_prefix("rr")
        .filter(|component| !component.is_empty())
        .unwrap_or(component);

    split_component_vowel_key(component)
}

fn split_component_vowel_key(component: &str) -> Option<&str> {
    for (boundary, _) in component.char_indices().skip(1) {
        let consonant = &component[..boundary];
        let vowel = &component[boundary..];

        if vowel_value(vowel).is_some()
            && (consonant_value(consonant).is_some() || is_phola_component_consonant(consonant))
        {
            return Some(vowel);
        }
    }

    None
}

fn is_phola_component_consonant(component: &str) -> bool {
    component == "w"
}

fn is_ya_consonant_unit(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant && matches!(unit.text.as_str(), "y" | "Y")
}

fn is_long_iya_marker(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Unknown && unit.text == "w"
}

fn promote_short_i_to_long_i(unit: &mut PhoneticUnit) {
    debug_assert!(is_short_i_vowel_bearing_unit(unit));
    unit.text.pop();
    unit.text.push('I');
}
