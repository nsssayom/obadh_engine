#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StemSuffixCompletion {
    pub text: String,
    pub suffix: &'static str,
    pub cost: u16,
}

#[derive(Debug, Clone, Copy)]
struct StemSuffixRule {
    text: &'static str,
    cost: u16,
}

const DETERMINER_SUFFIXES: &[StemSuffixRule] = &[
    rule("টি", 1),
    rule("টা", 1),
    rule("টুকু", 2),
    rule("খানা", 2),
    rule("খানি", 2),
    rule("টিকে", 2),
    rule("টাকে", 2),
    rule("টিতে", 2),
    rule("টাতে", 2),
    rule("টির", 2),
    rule("টার", 2),
    rule("টিও", 2),
    rule("টাও", 2),
    rule("টিই", 2),
    rule("টাই", 2),
];

const CASE_FOCUS_SUFFIXES: &[StemSuffixRule] = &[
    rule("কে", 1),
    rule("কেই", 2),
    rule("কেও", 2),
    rule("তে", 1),
    rule("তেই", 2),
    rule("তেও", 2),
    rule("র", 1),
    rule("ে", 1),
    rule("ের", 1),
    rule("ই", 1),
    rule("ও", 1),
];

const PLURAL_SUFFIXES: &[StemSuffixRule] = &[
    rule("রা", 1),
    rule("দের", 2),
    rule("দেরকে", 3),
    rule("গুলো", 2),
    rule("গুলা", 2),
    rule("গুলি", 2),
    rule("গুলোকে", 3),
    rule("গুলিকে", 3),
    rule("গুলোতে", 3),
    rule("গুলিতে", 3),
    rule("গুলোর", 3),
    rule("গুলির", 3),
    rule("গুলোও", 3),
    rule("গুলিও", 3),
];

const STEM_SUFFIX_GROUPS: &[&[StemSuffixRule]] =
    &[DETERMINER_SUFFIXES, CASE_FOCUS_SUFFIXES, PLURAL_SUFFIXES];

const fn rule(text: &'static str, cost: u16) -> StemSuffixRule {
    StemSuffixRule { text, cost }
}

pub fn stem_suffix_completions(stem: &str) -> impl Iterator<Item = StemSuffixCompletion> + '_ {
    let groups: &[&[StemSuffixRule]] = if stem.is_empty() || !is_bangla_surface_word(stem) {
        &[]
    } else {
        STEM_SUFFIX_GROUPS
    };

    groups
        .iter()
        .flat_map(|group| group.iter())
        .map(move |suffix| StemSuffixCompletion {
            text: suffixed_text(stem, suffix.text),
            suffix: suffix.text,
            cost: suffix.cost,
        })
        .filter(move |completion| completion.text != stem)
}

#[cfg(test)]
fn stem_suffix_rule_count() -> usize {
    STEM_SUFFIX_GROUPS.iter().map(|group| group.len()).sum()
}

#[cfg(test)]
fn stem_suffix_surfaces(stem: &str) -> Vec<StemSuffixCompletion> {
    stem_suffix_completions(stem).collect()
}

fn suffixed_text(stem: &str, suffix: &str) -> String {
    let mut text = String::with_capacity(stem.len() + suffix.len());
    text.push_str(stem);
    text.push_str(suffix);
    text
}

fn is_bangla_surface_word(text: &str) -> bool {
    text.chars().all(|ch| matches!(ch, '\u{0980}'..='\u{09FF}'))
}

#[cfg(test)]
mod tests {
    use super::{stem_suffix_rule_count, stem_suffix_surfaces};

    #[test]
    fn generates_bounded_suffix_surfaces_for_bangla_stems() {
        let completions = stem_suffix_surfaces("নদী");
        let texts = completions
            .iter()
            .map(|completion| completion.text.as_str())
            .collect::<Vec<_>>();

        assert!(texts.contains(&"নদীটি"));
        assert!(texts.contains(&"নদীকে"));
        assert!(texts.contains(&"নদীতে"));
        assert!(texts.contains(&"নদীর"));
        assert_eq!(completions.len(), stem_suffix_rule_count());
    }

    #[test]
    fn ignores_non_bangla_stems() {
        assert!(stem_suffix_surfaces("nodi").is_empty());
    }
}
