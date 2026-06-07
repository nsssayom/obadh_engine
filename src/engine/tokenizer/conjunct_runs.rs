use crate::definitions::conjuncts::ConjunctDefinitions;

use super::parts::BorrowedParts;
use super::{reph_base_part, PhoneticUnit, PhoneticUnitType};

pub(super) fn form_conjuncts_in_range(
    units: &mut [PhoneticUnit],
    start: usize,
    end: usize,
    conjunct_defs: &ConjunctDefinitions,
) {
    if end.saturating_sub(start) <= 1 {
        return;
    }

    let mut i = start;

    while i < end {
        if units[i].text.is_empty() {
            i += 1;
            continue;
        }

        if let Some(length) = longest_conjunct_prefix_in_range(units, i, end, conjunct_defs) {
            let conjunct_text = conjunct_text_for_range(units, i, length);

            let position = units[i].position;
            units[i] = PhoneticUnit {
                text: conjunct_text,
                unit_type: PhoneticUnitType::Conjunct,
                position,
            };

            for unit in units.iter_mut().take(i + length).skip(i + 1) {
                unit.text.clear();
            }
        }

        i += 1;
    }
}

pub(super) fn is_conjunct_run_component(unit: &PhoneticUnit) -> bool {
    matches!(
        unit.unit_type,
        PhoneticUnitType::Consonant
            | PhoneticUnitType::Conjunct
            | PhoneticUnitType::RephOverConsonant
    ) || (unit.unit_type == PhoneticUnitType::SpecialForm && unit.text == "rr")
        || (unit.unit_type == PhoneticUnitType::Unknown && unit.text == "w")
}

fn longest_conjunct_prefix_in_range(
    units: &[PhoneticUnit],
    start: usize,
    end: usize,
    conjunct_defs: &ConjunctDefinitions,
) -> Option<usize> {
    let mut node = conjunct_defs.conjunct_match_root();
    let mut best_length = None;

    'candidate: for (current, unit) in units.iter().enumerate().take(end).skip(start) {
        if unit.text.is_empty() {
            break;
        }

        for part in unit.text.split(",,") {
            let Some(next_node) = conjunct_defs.advance_conjunct_match(node, part) else {
                break 'candidate;
            };
            node = next_node;
        }

        let length = current - start + 1;
        if length >= 2 && conjunct_defs.conjunct_match_value(node).is_some() {
            best_length = Some(length);
        }
    }

    best_length.or_else(|| reph_tail_conjunct_prefix_in_range(units, start, end, conjunct_defs))
}

fn reph_tail_conjunct_prefix_in_range(
    units: &[PhoneticUnit],
    start: usize,
    end: usize,
    conjunct_defs: &ConjunctDefinitions,
) -> Option<usize> {
    let first = units.get(start)?;
    let reph_base = reph_base_part(first)?;
    let mut tail_parts = BorrowedParts::from_one(reph_base);
    let mut best_length = None;

    for (current, unit) in units.iter().enumerate().take(end).skip(start + 1) {
        if unit.text.is_empty() {
            break;
        }

        for part in unit.text.split(",,") {
            tail_parts.push(part);
        }

        let length = current - start + 1;
        if tail_parts.len() >= 2
            && conjunct_defs.can_form_conjunct_from_parts(tail_parts.as_slice())
            && !is_ambiguous_reph_r_phola_before_vowel(units, start, length, tail_parts.as_slice())
        {
            best_length = Some(length);
        }
    }

    best_length
}

fn is_ambiguous_reph_r_phola_before_vowel(
    units: &[PhoneticUnit],
    start: usize,
    length: usize,
    tail_parts: &[&str],
) -> bool {
    tail_parts.last() == Some(&"r")
        && units.get(start + length).is_some_and(|unit| {
            matches!(
                unit.unit_type,
                PhoneticUnitType::Vowel | PhoneticUnitType::TerminatingVowel
            )
        })
}

fn conjunct_text_for_range(units: &[PhoneticUnit], start: usize, length: usize) -> String {
    let mut conjunct_text = String::new();

    for unit in &units[start..start + length] {
        push_conjunct_text_parts(&mut conjunct_text, unit);
    }

    conjunct_text
}

fn push_conjunct_text_parts(conjunct_text: &mut String, unit: &PhoneticUnit) {
    if let Some(reph_base) = reph_base_part(unit) {
        push_conjunct_text_part(conjunct_text, "rr");
        push_conjunct_text_part(conjunct_text, reph_base);
        return;
    }

    for part in unit.text.split(",,") {
        push_conjunct_text_part(conjunct_text, part);
    }
}

fn push_conjunct_text_part(conjunct_text: &mut String, part: &str) {
    if !conjunct_text.is_empty() {
        conjunct_text.push_str(",,");
    }
    conjunct_text.push_str(part);
}
