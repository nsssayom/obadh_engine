use std::collections::BTreeSet;

const MAX_ROMAN_EDIT_INPUT_CHARS: usize = 32;
const MAX_ROMAN_EDIT_OUTPUTS: usize = 24;
const LATE_SINGLE_VOWELS: [char; 4] = ['e', 'a', 'i', 'u'];
const TWO_GAP_PRIORITY: [(char, char); 15] = [
    ('e', 'o'),
    ('O', 'a'),
    ('o', 'a'),
    ('O', 'e'),
    ('o', 'e'),
    ('o', 'o'),
    ('a', 'o'),
    ('e', 'a'),
    ('a', 'a'),
    ('e', 'e'),
    ('a', 'e'),
    ('O', 'o'),
    ('i', 'o'),
    ('u', 'o'),
    ('i', 'a'),
];
const TWO_GAP_FINAL_PRIORITY: [(char, char, char); 8] = [
    ('O', 'a', 'e'),
    ('o', 'a', 'e'),
    ('O', 'a', 'O'),
    ('e', 'o', 'e'),
    ('o', 'e', 'e'),
    ('o', 'o', 'e'),
    ('a', 'o', 'e'),
    ('o', 'a', 'o'),
];

pub(crate) fn roman_edit_candidates(
    roman_input: &str,
    obadh_output: &str,
    mut transliterate: impl FnMut(&str) -> String,
) -> Vec<String> {
    let chars = roman_input.chars().collect::<Vec<_>>();
    if chars.len() > MAX_ROMAN_EDIT_INPUT_CHARS || !is_simple_roman_word(&chars) {
        return Vec::new();
    }

    let mut generator = RomanEditGenerator {
        obadh_output,
        seen_bangla: BTreeSet::new(),
        outputs: Vec::new(),
        transliterate: &mut transliterate,
    };
    generator.push_candidates(&chars);
    generator.outputs
}

struct RomanEditGenerator<'a, F>
where
    F: FnMut(&str) -> String,
{
    obadh_output: &'a str,
    seen_bangla: BTreeSet<String>,
    outputs: Vec<String>,
    transliterate: &'a mut F,
}

impl<F> RomanEditGenerator<'_, F>
where
    F: FnMut(&str) -> String,
{
    fn push_candidates(&mut self, chars: &[char]) {
        for alias_chars in roman_alias_variants(chars) {
            if self.push_roman_base_outputs(&alias_chars, true) {
                return;
            }

            if is_sparse_roman_vowel_input(&alias_chars) {
                for variant in roman_o_to_o_variants(&alias_chars) {
                    if self.push_roman_base_outputs(&variant, true) {
                        return;
                    }
                }
            }
        }

        if is_sparse_roman_vowel_input(chars) {
            for variant in roman_o_to_o_variants(chars) {
                if self.push_roman_base_outputs(&variant, true) {
                    return;
                }
            }
        }

        self.push_roman_missing_vowel_outputs(chars);
    }

    fn push_roman_base_outputs(&mut self, chars: &[char], include_direct: bool) -> bool {
        if include_direct {
            let roman = roman_chars_to_string(chars);
            self.push_roman_edit_output(&roman);
            if self.is_full() {
                return true;
            }
        }

        self.push_roman_missing_vowel_outputs(chars);
        self.is_full()
    }

    fn push_roman_missing_vowel_outputs(&mut self, chars: &[char]) {
        let insertion_gaps = roman_missing_vowel_gaps(chars);
        let sparse_vowel_input = is_sparse_roman_vowel_input(chars);

        for &gap in &insertion_gaps {
            let variant = roman_variant_with_insertions(chars, &[(gap, 'o')]);
            self.push_roman_edit_output(&variant);
            if self.is_full() {
                return;
            }
        }

        if !sparse_vowel_input {
            return;
        }

        for &gap in &insertion_gaps {
            let variant = roman_variant_with_insertions(chars, &[(gap, 'O')]);
            self.push_roman_edit_output(&variant);
            if self.is_full() {
                return;
            }
        }

        for first_gap_index in 0..insertion_gaps.len() {
            for second_gap_index in first_gap_index + 1..insertion_gaps.len() {
                for &(first_vowel, second_vowel) in &TWO_GAP_PRIORITY {
                    let insertions = [
                        (insertion_gaps[first_gap_index], first_vowel),
                        (insertion_gaps[second_gap_index], second_vowel),
                    ];
                    let variant = roman_variant_with_insertions(chars, &insertions);
                    self.push_roman_edit_output(&variant);
                    if self.is_full() {
                        return;
                    }
                }
            }
        }

        if chars.last().is_some_and(|ch| is_roman_consonant(*ch)) {
            for first_gap_index in 0..insertion_gaps.len() {
                for second_gap_index in first_gap_index + 1..insertion_gaps.len() {
                    for &(first_vowel, second_vowel, final_vowel) in &TWO_GAP_FINAL_PRIORITY {
                        let insertions = [
                            (insertion_gaps[first_gap_index], first_vowel),
                            (insertion_gaps[second_gap_index], second_vowel),
                            (chars.len(), final_vowel),
                        ];
                        let variant = roman_variant_with_insertions(chars, &insertions);
                        self.push_roman_edit_output(&variant);
                        if self.is_full() {
                            return;
                        }
                    }
                }
            }
        }

        for &vowel in &LATE_SINGLE_VOWELS {
            for &gap in &insertion_gaps {
                let variant = roman_variant_with_insertions(chars, &[(gap, vowel)]);
                self.push_roman_edit_output(&variant);
                if self.is_full() {
                    return;
                }
            }
        }
    }

    fn push_roman_edit_output(&mut self, variant: &str) {
        let output = (self.transliterate)(variant);
        if output != self.obadh_output && self.seen_bangla.insert(output.clone()) {
            self.outputs.push(output);
        }
    }

    fn is_full(&self) -> bool {
        self.outputs.len() >= MAX_ROMAN_EDIT_OUTPUTS
    }
}

fn roman_missing_vowel_gaps(chars: &[char]) -> Vec<usize> {
    let mut gaps = Vec::new();

    for index in 1..chars.len() {
        if !is_roman_consonant(chars[index - 1]) || !is_roman_consonant(chars[index]) {
            continue;
        }
        if chars[index - 1].eq_ignore_ascii_case(&chars[index]) {
            continue;
        }
        gaps.push(index);
    }

    gaps
}

fn roman_alias_variants(chars: &[char]) -> Vec<Vec<char>> {
    let mut variants = Vec::new();

    for (index, ch) in chars.iter().copied().enumerate() {
        for alias in roman_aliases(ch) {
            let mut variant = chars.to_vec();
            variant[index] = *alias;
            variants.push(variant);
        }
    }

    variants
}

fn roman_aliases(ch: char) -> &'static [char] {
    match ch {
        'j' => &['z'],
        _ => &[],
    }
}

fn roman_o_to_o_variants(chars: &[char]) -> Vec<Vec<char>> {
    let mut variants = Vec::new();

    for (index, ch) in chars.iter().copied().enumerate() {
        if ch != 'o' {
            continue;
        }
        let mut variant = chars.to_vec();
        variant[index] = 'O';
        variants.push(variant);
    }

    variants
}

fn roman_chars_to_string(chars: &[char]) -> String {
    chars.iter().collect()
}

fn roman_variant_with_insertions(chars: &[char], insertions: &[(usize, char)]) -> String {
    let mut variant = String::with_capacity(chars.len() + insertions.len());
    let mut insertion_index = 0;

    for (char_index, ch) in chars.iter().enumerate() {
        while insertion_index < insertions.len() && insertions[insertion_index].0 == char_index {
            variant.push(insertions[insertion_index].1);
            insertion_index += 1;
        }
        variant.push(*ch);
    }

    while insertion_index < insertions.len() && insertions[insertion_index].0 == chars.len() {
        variant.push(insertions[insertion_index].1);
        insertion_index += 1;
    }

    variant
}

fn is_simple_roman_word(chars: &[char]) -> bool {
    chars
        .iter()
        .all(|ch| ch.is_ascii_alphabetic() || matches!(ch, '\'' | '`' | ',' | '.'))
}

fn is_roman_consonant(ch: char) -> bool {
    let ch = ch.to_ascii_lowercase();
    ch.is_ascii_alphabetic() && !is_roman_vowel(ch)
}

fn is_sparse_roman_vowel_input(chars: &[char]) -> bool {
    chars.iter().filter(|ch| is_roman_vowel(**ch)).count() <= 1
}

fn is_roman_vowel(ch: char) -> bool {
    matches!(ch.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
}
