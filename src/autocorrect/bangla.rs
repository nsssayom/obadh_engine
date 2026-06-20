#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitClass {
    Other,
    VowelSign,
    Hasant,
    NasalMark,
}

pub(crate) fn bangla_units(text: &str) -> Vec<&str> {
    let mut units = Vec::new();
    let mut start: Option<usize> = None;
    let mut end = 0;
    let mut join_next = false;

    for (index, ch) in text.char_indices() {
        let class = unit_class(ch);
        if start.is_none() {
            start = Some(index);
            end = index + ch.len_utf8();
            join_next = class == UnitClass::Hasant;
            continue;
        }

        if class != UnitClass::Other || join_next {
            end = index + ch.len_utf8();
            join_next = class == UnitClass::Hasant;
            continue;
        }

        let unit_start = start.expect("unit start should be set");
        units.push(&text[unit_start..end]);
        start = Some(index);
        end = index + ch.len_utf8();
        join_next = false;
    }

    if let Some(unit_start) = start {
        units.push(&text[unit_start..end]);
    }

    units
}

pub(crate) fn unit_class(ch: char) -> UnitClass {
    match ch {
        '\u{0981}'..='\u{0983}' => UnitClass::NasalMark,
        '\u{09BC}' => UnitClass::VowelSign,
        '\u{09BE}'..='\u{09C4}' => UnitClass::VowelSign,
        '\u{09C7}'..='\u{09C8}' => UnitClass::VowelSign,
        '\u{09CB}'..='\u{09CC}' => UnitClass::VowelSign,
        '\u{09CD}' => UnitClass::Hasant,
        '\u{09D7}' => UnitClass::VowelSign,
        '\u{09E2}'..='\u{09E3}' => UnitClass::VowelSign,
        _ => UnitClass::Other,
    }
}

pub(crate) fn unit_similarity(left: &str, right: &str) -> u16 {
    if left == right {
        return 0;
    }

    let left_tail = left.chars().last().map(unit_class);
    let right_tail = right.chars().last().map(unit_class);
    if left_tail == right_tail && matches!(left_tail, Some(UnitClass::VowelSign)) {
        return 1;
    }
    if matches!(left_tail, Some(UnitClass::NasalMark))
        || matches!(right_tail, Some(UnitClass::NasalMark))
    {
        return 1;
    }
    if left.contains('\u{09CD}') || right.contains('\u{09CD}') {
        return 2;
    }

    3
}

pub(crate) fn phonetic_skeleton(text: &str) -> String {
    let mut skeleton = String::with_capacity(text.len());
    let mut previous: Option<char> = None;

    for ch in text.chars() {
        let Some(code) = skeleton_char(ch) else {
            continue;
        };
        if previous == Some(code) {
            continue;
        }
        skeleton.push(code);
        previous = Some(code);
    }

    skeleton
}

fn skeleton_char(ch: char) -> Option<char> {
    if is_independent_vowel(ch) {
        return None;
    }

    match ch {
        '\u{0981}'..='\u{0983}' => Some('ং'),
        '\u{0995}'..='\u{09B9}'
        | '\u{09CE}'
        | '\u{09DC}'..='\u{09DD}'
        | '\u{09DF}'..='\u{09E1}' => Some(fold_base_consonant(ch)),
        _ if matches!(unit_class(ch), UnitClass::VowelSign | UnitClass::Hasant) => None,
        _ => None,
    }
}

fn is_independent_vowel(ch: char) -> bool {
    matches!(
        ch,
        '\u{0985}'..='\u{098C}' | '\u{098F}'..='\u{0990}' | '\u{0993}'..='\u{0994}'
    )
}

fn fold_base_consonant(ch: char) -> char {
    let ch = fold_aspiration(ch).unwrap_or(ch);
    match ch {
        'ঙ' | 'ঞ' | 'ণ' | 'ন' => 'ন',
        'শ' | 'ষ' | 'স' => 'স',
        'য' | 'য়' => 'য',
        'ড়' | 'ঢ়' => 'ড',
        _ => ch,
    }
}

fn fold_aspiration(ch: char) -> Option<char> {
    const VARGA_STARTS: [u32; 5] = [0x0995, 0x099A, 0x099F, 0x09A4, 0x09AA];

    let codepoint = ch as u32;
    for start in VARGA_STARTS {
        if !(start..=start + 4).contains(&codepoint) {
            continue;
        }

        let offset = codepoint - start;
        let folded_offset = match offset {
            1 => 0,
            3 => 2,
            _ => offset,
        };
        return char::from_u32(start + folded_offset);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{bangla_units, phonetic_skeleton};

    #[test]
    fn bangla_units_group_vowel_signs_and_conjuncts() {
        assert_eq!(bangla_units("কিরণ"), vec!["কি", "র", "ণ"]);
        assert_eq!(bangla_units("বিজ্ঞান"), vec!["বি", "জ্ঞা", "ন"]);
    }

    #[test]
    fn phonetic_skeleton_is_consonant_heavy_and_folded() {
        assert_eq!(phonetic_skeleton("কিরণ"), "করন");
        assert_eq!(phonetic_skeleton("করণ"), "করন");
        assert_eq!(phonetic_skeleton("শাসন"), "সন");
        assert_eq!(phonetic_skeleton("বিজ্ঞান"), "বজন");
        assert_eq!(phonetic_skeleton("অঞ্চল"), phonetic_skeleton("আনছল"));
        assert_eq!(phonetic_skeleton("অক্ষর"), phonetic_skeleton("আক্ষার"));
    }
}
