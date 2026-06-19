use std::sync::OnceLock;

use crate::definitions::{
    consonant_categories, diacritic_rules, diacritics::HASANT, symbol_rules, vowel_rules,
};

use super::PhoneticUnitType;

pub(super) struct RuleMatch<'a> {
    pub(super) text: &'static str,
    pub(super) byte_len: usize,
    pub(super) unit_type: PhoneticUnitType,
    _marker: std::marker::PhantomData<&'a str>,
}

struct PhoneticPattern {
    input: &'static str,
    canonical: &'static str,
    unit_type: PhoneticUnitType,
}

#[derive(Default)]
pub(super) struct PatternTrie {
    nodes: Vec<TrieNode>,
    case_fallbacks: Vec<CaseFallback>,
}

#[derive(Default)]
struct TrieNode {
    terminal: Option<TrieTerminal>,
    edges: Vec<TrieEdge>,
}

#[derive(Clone, Copy)]
struct TrieTerminal {
    canonical: &'static str,
    unit_type: PhoneticUnitType,
}

#[derive(Clone, Copy)]
struct CaseFallback {
    input: u8,
    canonical: &'static str,
    unit_type: PhoneticUnitType,
}

#[derive(Clone, Copy)]
struct TrieEdge {
    byte: u8,
    node: usize,
}

impl PatternTrie {
    fn from_patterns(patterns: &[PhoneticPattern]) -> Self {
        let mut trie = Self {
            nodes: Vec::with_capacity(pattern_trie_node_capacity(patterns)),
            case_fallbacks: case_fallback_rules_for(patterns),
        };
        trie.nodes.push(TrieNode::default());

        for pattern in patterns {
            trie.insert(pattern);
        }

        for node in &mut trie.nodes {
            node.edges.sort_unstable_by_key(|edge| edge.byte);
        }
        trie.case_fallbacks
            .sort_unstable_by_key(|fallback| fallback.input);

        trie
    }

    fn insert(&mut self, pattern: &PhoneticPattern) {
        let mut node = 0;

        for byte in pattern.input.bytes() {
            node = self.child_or_insert(node, byte);
        }

        assert!(
            self.nodes[node].terminal.is_none(),
            "duplicate phonetic pattern: {}",
            pattern.input
        );
        self.nodes[node].terminal = Some(TrieTerminal {
            canonical: pattern.canonical,
            unit_type: pattern.unit_type,
        });
    }

    fn child_or_insert(&mut self, node: usize, byte: u8) -> usize {
        if let Some(edge) = self.nodes[node].edges.iter().find(|edge| edge.byte == byte) {
            return edge.node;
        }

        let child = self.nodes.len();
        self.nodes.push(TrieNode::default());
        self.nodes[node].edges.push(TrieEdge { byte, node: child });
        child
    }

    #[inline]
    pub(super) fn match_at<'a>(&self, text: &'a str, start: usize) -> Option<RuleMatch<'a>> {
        let mut node = 0;
        let mut best_match = None;

        for (offset, byte) in text.as_bytes().get(start..)?.iter().copied().enumerate() {
            let Some(next_node) = self.nodes[node].child(byte) else {
                break;
            };
            node = next_node;
            let end = start + offset + 1;

            if let Some(terminal) = self.nodes[node].terminal {
                best_match = Some(RuleMatch {
                    text: terminal.canonical,
                    byte_len: end - start,
                    unit_type: terminal.unit_type,
                    _marker: std::marker::PhantomData,
                });
            }
        }

        best_match.or_else(|| self.case_fallback_at(text, start))
    }

    #[inline]
    fn case_fallback_at<'a>(&self, text: &'a str, start: usize) -> Option<RuleMatch<'a>> {
        let byte = *text.as_bytes().get(start)?;
        if !byte.is_ascii_alphabetic() {
            return None;
        }
        let fallback = self
            .case_fallbacks
            .binary_search_by_key(&byte, |fallback| fallback.input)
            .ok()
            .map(|index| self.case_fallbacks[index])?;

        Some(RuleMatch {
            text: fallback.canonical,
            byte_len: 1,
            unit_type: fallback.unit_type,
            _marker: std::marker::PhantomData,
        })
    }
}

impl TrieNode {
    #[inline]
    fn child(&self, byte: u8) -> Option<usize> {
        self.edges
            .binary_search_by_key(&byte, |edge| edge.byte)
            .ok()
            .map(|index| self.edges[index].node)
    }
}

pub(super) fn phonetic_pattern_trie_static() -> &'static PatternTrie {
    static INSTANCE: OnceLock<PatternTrie> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let patterns = phonetic_patterns();
        PatternTrie::from_patterns(&patterns)
    })
}

fn phonetic_patterns() -> Vec<PhoneticPattern> {
    let mut patterns = Vec::with_capacity(phonetic_pattern_count());
    append_exact_phonetic_patterns(&mut patterns);
    patterns
}

fn append_exact_phonetic_patterns(patterns: &mut Vec<PhoneticPattern>) {
    if vowel_rules().iter().any(|(roman, _)| *roman == "o") {
        push_exact(patterns, "o", PhoneticUnitType::TerminatingVowel);
    }

    push_exact(patterns, "rr", PhoneticUnitType::SpecialForm);

    for &(roman, bengali) in diacritic_rules() {
        let unit_type = if bengali == HASANT {
            PhoneticUnitType::ConsonantWithHasant
        } else {
            PhoneticUnitType::SpecialForm
        };
        push_exact(patterns, roman, unit_type);
    }

    for &(roman, _) in symbol_rules() {
        push_exact(patterns, roman, PhoneticUnitType::Symbol);
    }

    for category in consonant_categories() {
        for &(roman, _) in category {
            push_exact(patterns, roman, PhoneticUnitType::Consonant);
        }
    }

    for &(roman, _) in vowel_rules() {
        if roman != "o" {
            push_exact(patterns, roman, PhoneticUnitType::Vowel);
        }
    }
}

fn push_exact(
    patterns: &mut Vec<PhoneticPattern>,
    input: &'static str,
    unit_type: PhoneticUnitType,
) {
    patterns.push(exact_pattern(input, unit_type));
}

fn exact_pattern(input: &'static str, unit_type: PhoneticUnitType) -> PhoneticPattern {
    PhoneticPattern {
        input,
        canonical: input,
        unit_type,
    }
}

fn case_fallback_rules_for(exact_patterns: &[PhoneticPattern]) -> Vec<CaseFallback> {
    let mut fallbacks = Vec::new();
    for pattern in exact_patterns {
        if pattern.input != pattern.canonical {
            continue;
        }
        let Some(input) = opposite_case_input_for(pattern.input.as_ref()) else {
            continue;
        };
        if is_reserved_case_fallback_input(input)
            || has_exact_pattern_input(exact_patterns, input)
            || has_case_fallback_input(&fallbacks, input)
        {
            continue;
        }

        fallbacks.push(CaseFallback {
            input,
            canonical: pattern.canonical,
            unit_type: pattern.unit_type,
        });
    }
    fallbacks
}

fn opposite_case_input_for(input: &str) -> Option<u8> {
    if input.len() != 1 {
        return None;
    }
    let byte = input.as_bytes()[0];
    if byte.is_ascii_lowercase() {
        Some(byte.to_ascii_uppercase())
    } else if byte.is_ascii_uppercase() {
        Some(byte.to_ascii_lowercase())
    } else {
        None
    }
}

fn is_reserved_case_fallback_input(input: u8) -> bool {
    // `Z` is deliberately consumed as an unknown marker by the narrow `rZy`
    // non-conjunct ra-ya normalization path. It must not become a generic `z`.
    input == b'Z'
}

fn has_exact_pattern_input(patterns: &[PhoneticPattern], input: u8) -> bool {
    patterns
        .iter()
        .any(|pattern| pattern.input.as_bytes() == [input])
}

fn has_case_fallback_input(fallbacks: &[CaseFallback], input: u8) -> bool {
    fallbacks.iter().any(|fallback| fallback.input == input)
}

fn phonetic_pattern_count() -> usize {
    exact_phonetic_pattern_count()
}

fn exact_phonetic_pattern_count() -> usize {
    let has_terminating_o = vowel_rules().iter().any(|(roman, _)| *roman == "o") as usize;
    let vowel_count_without_terminating_o = vowel_rules()
        .iter()
        .filter(|(roman, _)| *roman != "o")
        .count();
    let consonant_count = consonant_categories()
        .into_iter()
        .map(<[_]>::len)
        .sum::<usize>();

    has_terminating_o
        + 1
        + diacritic_rules().len()
        + symbol_rules().len()
        + consonant_count
        + vowel_count_without_terminating_o
}

#[cfg(test)]
fn case_fallback_pattern_count() -> usize {
    let mut patterns = Vec::with_capacity(exact_phonetic_pattern_count());
    append_exact_phonetic_patterns(&mut patterns);
    case_fallback_rules_for(&patterns).len()
}

fn pattern_trie_node_capacity(patterns: &[PhoneticPattern]) -> usize {
    1 + patterns
        .iter()
        .map(|pattern| pattern.input.len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_trie_uses_longest_prefix_match() {
        let trie = PatternTrie::from_patterns(&[
            exact_pattern("rr", PhoneticUnitType::SpecialForm),
            exact_pattern("r", PhoneticUnitType::Consonant),
            exact_pattern("rri", PhoneticUnitType::Vowel),
        ]);

        let rule_match = trie.match_at("rria", 0).expect("rri should match");
        assert_eq!(rule_match.text, "rri");
        assert_eq!(rule_match.byte_len, 3);
        assert_eq!(rule_match.unit_type, PhoneticUnitType::Vowel);
    }

    #[test]
    #[should_panic(expected = "duplicate phonetic pattern: aa")]
    fn pattern_trie_rejects_duplicate_patterns() {
        let _ = PatternTrie::from_patterns(&[
            exact_pattern("aa", PhoneticUnitType::Vowel),
            PhoneticPattern {
                input: "aa",
                canonical: "a",
                unit_type: PhoneticUnitType::Consonant,
            },
        ]);
    }

    #[test]
    fn phonetic_patterns_reserve_exact_rule_count() {
        let patterns = phonetic_patterns();

        assert_eq!(patterns.len(), phonetic_pattern_count());
        assert_eq!(patterns.capacity(), phonetic_pattern_count());
    }

    #[test]
    fn pattern_trie_presizes_maximum_node_count() {
        let patterns = phonetic_patterns();
        let trie = PatternTrie::from_patterns(&patterns);

        assert!(trie.nodes.capacity() >= pattern_trie_node_capacity(&patterns));
    }

    #[test]
    fn case_fallback_patterns_are_collision_checked() {
        let patterns = phonetic_patterns();
        let fallback_rules = case_fallback_rules_for(&patterns);

        assert_eq!(fallback_rules.len(), case_fallback_pattern_count());

        for fallback in fallback_rules {
            assert_eq!(fallback.canonical.len(), 1);
            assert_eq!(
                opposite_case_input_for(fallback.canonical),
                Some(fallback.input)
            );

            assert!(
                !has_exact_pattern_input(&patterns, fallback.input),
                "case fallback {} must not shadow an exact rule",
                fallback.input as char
            );
            assert!(
                patterns.iter().any(|pattern| {
                    pattern.input == fallback.canonical && pattern.unit_type == fallback.unit_type
                }),
                "case fallback {} must point to an exact canonical rule",
                fallback.input as char
            );
        }
    }

    #[test]
    fn case_fallback_generation_is_symmetric_for_unclaimed_case() {
        let patterns = vec![
            exact_pattern("X", PhoneticUnitType::Consonant),
            exact_pattern("a", PhoneticUnitType::Vowel),
            exact_pattern("A", PhoneticUnitType::Vowel),
        ];

        let trie = PatternTrie::from_patterns(&patterns);
        let fallback = trie.match_at("x", 0).expect("x should fallback to X");

        assert_eq!(fallback.text, "X");
        assert_eq!(fallback.byte_len, 1);
        assert_eq!(fallback.unit_type, PhoneticUnitType::Consonant);
        assert_eq!(
            trie.match_at("a", 0).expect("exact a should match").text,
            "a"
        );
    }

    #[test]
    fn case_fallback_patterns_do_not_claim_reserved_uppercase_signals() {
        let patterns = phonetic_patterns();
        let fallback_inputs = case_fallback_rules_for(&patterns)
            .into_iter()
            .map(|fallback| fallback.input as char)
            .collect::<std::collections::BTreeSet<_>>();

        for reserved in [
            'A', 'C', 'D', 'E', 'I', 'J', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'Y', 'Z',
        ] {
            assert!(
                !fallback_inputs.contains(&reserved),
                "reserved uppercase signal {reserved} must not become a case fallback"
            );
        }
    }
}
