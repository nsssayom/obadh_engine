use super::bangla::{bangla_units, unit_similarity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EditCost(pub u16);

pub(crate) const INSERT_DELETE_COST: u16 = 2;

pub fn weighted_edit_distance(left: &str, right: &str) -> EditCost {
    let left_units = bangla_units(left);
    let right_units = bangla_units(right);
    weighted_unit_edit_distance(&left_units, &right_units)
}

pub(crate) fn weighted_unit_edit_distance(left: &[&str], right: &[&str]) -> EditCost {
    if left.is_empty() {
        return EditCost((right.len() as u16) * INSERT_DELETE_COST);
    }
    if right.is_empty() {
        return EditCost((left.len() as u16) * INSERT_DELETE_COST);
    }

    let mut previous = (0..=right.len())
        .map(|index| (index as u16) * INSERT_DELETE_COST)
        .collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];

    for (left_index, left_unit) in left.iter().enumerate() {
        current[0] = ((left_index + 1) as u16) * INSERT_DELETE_COST;
        for (right_index, right_unit) in right.iter().enumerate() {
            let substitution = previous[right_index] + unit_similarity(left_unit, right_unit);
            let deletion = previous[right_index + 1] + INSERT_DELETE_COST;
            let insertion = current[right_index] + INSERT_DELETE_COST;
            current[right_index + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    EditCost(previous[right.len()])
}

#[cfg(test)]
mod tests {
    use super::weighted_edit_distance;

    #[test]
    fn weighted_edit_distance_is_bangla_unit_aware() {
        assert_eq!(weighted_edit_distance("বিজ্ঞান", "বিজ্ঞান").0, 0);
        assert!(weighted_edit_distance("বিজান", "বিজ্ঞান").0 <= 2);
        assert!(weighted_edit_distance("কিরণ", "করণ").0 < weighted_edit_distance("আম", "বিজ্ঞান").0);
    }
}
