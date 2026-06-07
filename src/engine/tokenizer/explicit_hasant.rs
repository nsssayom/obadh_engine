use crate::definitions::conjuncts::ConjunctDefinitions;

use super::{parts::BorrowedParts, PhoneticUnit, PhoneticUnitType};

struct ExplicitHasantChain {
    end: usize,
    text: String,
    position: usize,
}

pub(super) fn collapse_explicit_hasant_chains(
    units: &mut Vec<PhoneticUnit>,
    conjunct_defs: &ConjunctDefinitions,
) {
    let Some(first_match) = first_explicit_hasant_chain(units, conjunct_defs) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if let Some(chain) = explicit_hasant_chain_at(units, read, conjunct_defs) {
            units[write] = PhoneticUnit {
                text: chain.text,
                unit_type: PhoneticUnitType::Conjunct,
                position: chain.position,
            };
            read = chain.end;
            write += 1;
            continue;
        }

        super::move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

fn first_explicit_hasant_chain(
    units: &[PhoneticUnit],
    conjunct_defs: &ConjunctDefinitions,
) -> Option<usize> {
    (0..units.len().saturating_sub(2))
        .find(|&index| explicit_hasant_chain_at(units, index, conjunct_defs).is_some())
}

fn explicit_hasant_chain_at(
    units: &[PhoneticUnit],
    index: usize,
    conjunct_defs: &ConjunctDefinitions,
) -> Option<ExplicitHasantChain> {
    if index + 2 >= units.len()
        || units[index + 1].unit_type != PhoneticUnitType::ConsonantWithHasant
    {
        return None;
    }

    let mut parts = explicit_hasant_chain_start_parts(&units[index])?;
    let position = units[index].position;
    let mut next = index + 1;
    let mut consumed_hasant = false;

    while next < units.len() && units[next].unit_type == PhoneticUnitType::ConsonantWithHasant {
        if next + 1 >= units.len() {
            break;
        }

        if let Some(part) =
            explicit_hasant_chain_next_part(parts.as_slice(), &units[next + 1], conjunct_defs)
        {
            parts.push(part);
            consumed_hasant = true;
            next += 2;
        } else {
            break;
        }
    }

    if consumed_hasant
        && parts.len() >= 2
        && explicit_hasant_chain_is_renderable(parts.as_slice(), conjunct_defs)
    {
        return Some(ExplicitHasantChain {
            end: next,
            text: join_explicit_hasant_parts(parts.as_slice()),
            position,
        });
    }

    None
}

fn join_explicit_hasant_parts(parts: &[&str]) -> String {
    let separator_len = ",,".len() * parts.len().saturating_sub(1);
    let mut text =
        String::with_capacity(parts.iter().map(|part| part.len()).sum::<usize>() + separator_len);

    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            text.push_str(",,");
        }
        text.push_str(part);
    }

    text
}

fn explicit_hasant_chain_start_parts(unit: &PhoneticUnit) -> Option<BorrowedParts<'_>> {
    match unit.unit_type {
        PhoneticUnitType::Consonant => Some(BorrowedParts::from_one(unit.text.as_str())),
        PhoneticUnitType::RephOverConsonant => {
            let base = super::reph_base_part(unit)?;
            Some(BorrowedParts::from_two("rr", base))
        }
        PhoneticUnitType::SpecialForm if unit.text == "rr" => Some(BorrowedParts::from_one("rr")),
        _ => None,
    }
}

fn explicit_hasant_chain_next_part<'a>(
    parts: &[&str],
    unit: &'a PhoneticUnit,
    conjunct_defs: &ConjunctDefinitions,
) -> Option<&'a str> {
    match unit.unit_type {
        PhoneticUnitType::Consonant if is_explicit_phola_marker(&unit.text) => {
            if BorrowedParts::extended_is_valid(parts, &unit.text, conjunct_defs) {
                Some(unit.text.as_str())
            } else {
                None
            }
        }
        PhoneticUnitType::Consonant => Some(unit.text.as_str()),
        PhoneticUnitType::Unknown if unit.text == "w" => {
            if BorrowedParts::extended_is_valid(parts, "w", conjunct_defs) {
                Some("w")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn explicit_hasant_chain_is_renderable(
    parts: &[&str],
    conjunct_defs: &ConjunctDefinitions,
) -> bool {
    if !parts.iter().any(|part| is_explicit_phola_marker(part)) {
        return true;
    }

    conjunct_defs.can_form_conjunct_from_parts(parts)
}

fn is_explicit_phola_marker(part: &str) -> bool {
    matches!(part, "w" | "y" | "Y")
}
