use std::sync::OnceLock;

use crate::definitions::{
    consonant_categories, diacritic_rules, diacritics::HASANT, symbol_rules, vowel_rules,
};

use super::PhoneticUnitType;

pub(super) struct RuleMatch<'a> {
    pub(super) text: &'a str,
    pub(super) unit_type: PhoneticUnitType,
}

#[derive(Default)]
pub(super) struct PatternTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Default)]
struct TrieNode {
    unit_type: Option<PhoneticUnitType>,
    edges: Vec<TrieEdge>,
}

#[derive(Clone, Copy)]
struct TrieEdge {
    byte: u8,
    node: usize,
}

impl PatternTrie {
    fn from_patterns(patterns: &[(&'static str, PhoneticUnitType)]) -> Self {
        let mut trie = Self {
            nodes: Vec::with_capacity(pattern_trie_node_capacity(patterns)),
        };
        trie.nodes.push(TrieNode::default());

        for &(pattern, unit_type) in patterns {
            trie.insert(pattern, unit_type);
        }

        for node in &mut trie.nodes {
            node.edges.sort_unstable_by_key(|edge| edge.byte);
        }

        trie
    }

    fn insert(&mut self, pattern: &'static str, unit_type: PhoneticUnitType) {
        let mut node = 0;

        for byte in pattern.bytes() {
            node = self.child_or_insert(node, byte);
        }

        assert!(
            self.nodes[node].unit_type.is_none(),
            "duplicate phonetic pattern: {pattern}"
        );
        self.nodes[node].unit_type = Some(unit_type);
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

            if let Some(unit_type) = self.nodes[node].unit_type {
                best_match = Some(RuleMatch {
                    text: &text[start..end],
                    unit_type,
                });
            }
        }

        best_match
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

fn phonetic_patterns() -> Vec<(&'static str, PhoneticUnitType)> {
    let mut patterns = Vec::with_capacity(phonetic_pattern_count());

    if vowel_rules().iter().any(|(roman, _)| *roman == "o") {
        patterns.push(("o", PhoneticUnitType::TerminatingVowel));
    }

    patterns.push(("rr", PhoneticUnitType::SpecialForm));

    for &(roman, bengali) in diacritic_rules() {
        let unit_type = if bengali == HASANT {
            PhoneticUnitType::ConsonantWithHasant
        } else {
            PhoneticUnitType::SpecialForm
        };
        patterns.push((roman, unit_type));
    }

    for &(roman, _) in symbol_rules() {
        patterns.push((roman, PhoneticUnitType::Symbol));
    }

    for category in consonant_categories() {
        for &(roman, _) in category {
            patterns.push((roman, PhoneticUnitType::Consonant));
        }
    }

    for &(roman, _) in vowel_rules() {
        if roman != "o" {
            patterns.push((roman, PhoneticUnitType::Vowel));
        }
    }

    patterns
}

fn phonetic_pattern_count() -> usize {
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

fn pattern_trie_node_capacity(patterns: &[(&'static str, PhoneticUnitType)]) -> usize {
    1 + patterns
        .iter()
        .map(|(pattern, _)| pattern.len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_trie_uses_longest_prefix_match() {
        let trie = PatternTrie::from_patterns(&[
            ("rr", PhoneticUnitType::SpecialForm),
            ("r", PhoneticUnitType::Consonant),
            ("rri", PhoneticUnitType::Vowel),
        ]);

        let rule_match = trie.match_at("rria", 0).expect("rri should match");
        assert_eq!(rule_match.text, "rri");
        assert_eq!(rule_match.unit_type, PhoneticUnitType::Vowel);
    }

    #[test]
    #[should_panic(expected = "duplicate phonetic pattern: aa")]
    fn pattern_trie_rejects_duplicate_patterns() {
        let _ = PatternTrie::from_patterns(&[
            ("aa", PhoneticUnitType::Vowel),
            ("aa", PhoneticUnitType::Consonant),
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
}
