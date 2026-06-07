use super::{PhoneticUnit, PhoneticUnitType};

pub(super) struct TrailingDiacritics<'a> {
    text: &'a str,
    offset: usize,
}

impl TrailingDiacritics<'_> {
    pub(super) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

pub(super) fn split_trailing_diacritics(word: &str) -> (&str, TrailingDiacritics<'_>) {
    let mut base_end = word.len();

    for (position, marker) in word.char_indices().rev() {
        if !matches!(marker, '^' | ':') {
            break;
        }

        base_end = position;
    }

    (
        &word[..base_end],
        TrailingDiacritics {
            text: &word[base_end..],
            offset: base_end,
        },
    )
}

pub(super) fn append_trailing_diacritics(
    units: &mut Vec<PhoneticUnit>,
    suffix: TrailingDiacritics<'_>,
) {
    units.extend(
        suffix
            .text
            .char_indices()
            .map(|(offset, marker)| PhoneticUnit {
                text: marker.to_string(),
                unit_type: PhoneticUnitType::SpecialForm,
                position: suffix.offset + offset,
            }),
    );
}
