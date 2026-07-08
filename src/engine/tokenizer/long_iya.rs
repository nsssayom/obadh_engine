use crate::definitions::{consonant_value, vowel_value};

use super::{PhoneticUnit, PhoneticUnitType};

pub(super) fn normalize_iyw_long_iya_signal(units: &mut Vec<PhoneticUnit>) -> bool {
    let Some(first_match) = first_iyw_long_iya_signal(units) else {
        return false;
    };

    let mut read = first_match;
    let mut write = first_match;
    while read < units.len() {
        if is_iyw_long_iya_signal_at(units, read) {
            // The `w` marker may have already absorbed the following vowel (kiywo →
            // `ki`,`y`,`wo`). Capture it before dropping the marker so it can be
            // re-homed onto the ya (কীয়ো), matching the vowel-less case (তীয়).
            let marker_vowel = long_iya_marker_trailing_vowel(&units[read + 2]);
            promote_short_i_to_long_i(&mut units[read]);
            super::move_unit(units, read, write);
            super::move_unit(units, read + 1, write + 1);
            write += 2;
            if let Some((vowel_text, vowel_type, position)) = marker_vowel {
                units[write] = PhoneticUnit {
                    text: vowel_text,
                    unit_type: vowel_type,
                    position,
                };
                write += 1;
            }
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
    match unit.unit_type {
        // Bare `w` (তীয় from tiyw).
        PhoneticUnitType::Consonant => unit.text == "w",
        // `w` that has absorbed a following vowel (কীয়ো from kiywo).
        PhoneticUnitType::ConsonantWithVowel | PhoneticUnitType::ConsonantWithTerminator => {
            long_iya_marker_trailing_vowel(unit).is_some()
        }
        _ => false,
    }
}

/// The vowel a `w` long-ঈয় marker carries, if it absorbed one during compaction.
/// Returns the vowel text and its unit type so the caller can re-emit it.
fn long_iya_marker_trailing_vowel(
    unit: &PhoneticUnit,
) -> Option<(String, PhoneticUnitType, usize)> {
    let vowel_type = match unit.unit_type {
        PhoneticUnitType::ConsonantWithVowel => PhoneticUnitType::Vowel,
        PhoneticUnitType::ConsonantWithTerminator => PhoneticUnitType::TerminatingVowel,
        _ => return None,
    };
    let vowel = unit.text.strip_prefix('w').filter(|vowel| !vowel.is_empty())?;
    Some((vowel.to_string(), vowel_type, unit.position))
}

fn promote_short_i_to_long_i(unit: &mut PhoneticUnit) {
    debug_assert!(is_short_i_vowel_bearing_unit(unit));
    unit.text.pop();
    unit.text.push('I');
}
