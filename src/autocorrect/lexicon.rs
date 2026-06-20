use std::collections::BTreeMap;

use super::artifact::{push_u16, push_u32, ArtifactReader, LexiconArtifactError, LEXICON_MAGIC};
use super::bangla::{bangla_units, phonetic_skeleton, unit_similarity};
use super::edit::{weighted_edit_distance, EditCost, INSERT_DELETE_COST};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexiconEntry {
    pub word: String,
    pub frequency: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    entries: Vec<LexiconEntry>,
    trie: Vec<LexiconNode>,
    skeleton_index: Vec<SkeletonIndexEntry>,
    skeleton_delete_index: Vec<SkeletonIndexEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexiconStats {
    pub entries: usize,
    pub trie_nodes: usize,
    pub trie_edges: usize,
    pub skeleton_keys: usize,
    pub skeleton_delete_keys: usize,
}

#[derive(Debug, Clone, Default)]
struct LexiconNode {
    children: Vec<(String, usize)>,
    entry_index: Option<usize>,
    min_terminal_depth: u16,
    max_terminal_depth: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkeletonIndexEntry {
    key_hash: u64,
    entry_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LexiconMatch<'a> {
    pub entry: &'a LexiconEntry,
    pub edit_cost: EditCost,
}

impl Lexicon {
    pub fn new(entries: impl IntoIterator<Item = LexiconEntry>) -> Self {
        let mut merged = BTreeMap::<String, u32>::new();
        for entry in entries {
            if entry.word.is_empty() {
                continue;
            }
            merged
                .entry(entry.word)
                .and_modify(|frequency| *frequency = (*frequency).max(entry.frequency))
                .or_insert(entry.frequency);
        }
        let entries = merged
            .into_iter()
            .map(|(word, frequency)| LexiconEntry { word, frequency })
            .collect::<Vec<_>>();
        Self::from_sorted_entries(entries)
    }

    pub fn from_words(words: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::new(words.into_iter().map(|word| LexiconEntry {
            word: word.into(),
            frequency: 1,
        }))
    }

    pub fn entries(&self) -> &[LexiconEntry] {
        &self.entries
    }

    pub fn frequency(&self, word: &str) -> Option<u32> {
        self.entries
            .binary_search_by(|entry| entry.word.as_str().cmp(word))
            .ok()
            .map(|index| self.entries[index].frequency)
    }

    pub fn contains(&self, word: &str) -> bool {
        self.frequency(word).is_some()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> LexiconStats {
        LexiconStats {
            entries: self.entries.len(),
            trie_nodes: self.trie.len(),
            trie_edges: self.trie.iter().map(|node| node.children.len()).sum(),
            skeleton_keys: self.skeleton_index.len(),
            skeleton_delete_keys: self.skeleton_delete_index.len(),
        }
    }

    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self, LexiconArtifactError> {
        let mut reader = ArtifactReader::new(bytes);
        reader.read_magic()?;
        let count = reader.read_u32()? as usize;
        let mut entries = Vec::with_capacity(count);

        for index in 0..count {
            let frequency = reader.read_u32()?;
            let word_len = reader.read_u16()? as usize;
            let word = reader.read_word(word_len)?;

            if word.is_empty() {
                return Err(LexiconArtifactError::EmptyWord { index });
            }
            if let Some(previous) = entries.last() {
                let previous: &LexiconEntry = previous;
                match previous.word.as_str().cmp(word) {
                    std::cmp::Ordering::Equal => {
                        return Err(LexiconArtifactError::DuplicateWord { index });
                    }
                    std::cmp::Ordering::Greater => {
                        return Err(LexiconArtifactError::UnsortedWord { index });
                    }
                    std::cmp::Ordering::Less => {}
                }
            }

            entries.push(LexiconEntry::new(word, frequency));
        }

        reader.finish()?;
        Ok(Self::from_sorted_entries(entries))
    }

    pub fn to_compact_bytes(&self) -> Result<Vec<u8>, LexiconArtifactError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(LEXICON_MAGIC);
        let entry_count = u32::try_from(self.entries.len()).map_err(|_| {
            LexiconArtifactError::TooManyEntries {
                entries: self.entries.len(),
            }
        })?;
        push_u32(&mut bytes, entry_count);

        for entry in &self.entries {
            push_u32(&mut bytes, entry.frequency);
            let word_bytes = entry.word.as_bytes();
            let word_len =
                u16::try_from(word_bytes.len()).map_err(|_| LexiconArtifactError::WordTooLong {
                    bytes: word_bytes.len(),
                })?;
            push_u16(&mut bytes, word_len);
            bytes.extend_from_slice(word_bytes);
        }

        Ok(bytes)
    }

    pub(crate) fn find_within_edit_cost(
        &self,
        input: &str,
        max_edit_cost: u16,
    ) -> Vec<LexiconMatch<'_>> {
        if self.trie.is_empty() {
            return Vec::new();
        }

        let input_units = bangla_units(input);
        let length_slack = usize::from(max_edit_cost / INSERT_DELETE_COST);
        let min_candidate_depth = input_units.len().saturating_sub(length_slack);
        let max_candidate_depth = input_units.len() + length_slack;
        let initial_row = (0..=input_units.len())
            .map(|index| (index as u16) * INSERT_DELETE_COST)
            .collect::<Vec<_>>();
        let mut matches = Vec::new();
        let mut rows = vec![initial_row];

        for (unit, child_index) in &self.trie[0].children {
            self.find_within_edit_cost_at(
                *child_index,
                unit,
                &input_units,
                1,
                &mut rows,
                max_edit_cost,
                min_candidate_depth,
                max_candidate_depth,
                &mut matches,
            );
        }

        matches
    }

    pub(crate) fn find_by_phonetic_skeleton(
        &self,
        input: &str,
        max_results: usize,
    ) -> Vec<LexiconMatch<'_>> {
        if self.skeleton_index.is_empty() || max_results == 0 {
            return Vec::new();
        }

        let key = phonetic_skeleton(input);
        if key.is_empty() {
            return Vec::new();
        }

        let key_hash = stable_hash(&key);
        let start = self
            .skeleton_index
            .partition_point(|entry| entry.key_hash < key_hash);
        let mut matches = Vec::new();

        for indexed in self.skeleton_index[start..]
            .iter()
            .take_while(|entry| entry.key_hash == key_hash)
        {
            let entry = &self.entries[indexed.entry_index];
            if phonetic_skeleton(&entry.word) != key {
                continue;
            }
            matches.push(LexiconMatch {
                entry,
                edit_cost: weighted_edit_distance(input, &entry.word),
            });
            if matches.len() >= max_results {
                break;
            }
        }

        matches
    }

    pub(crate) fn find_by_fuzzy_phonetic_skeleton(
        &self,
        input: &str,
        max_skeleton_cost: u16,
        max_results: usize,
    ) -> Vec<LexiconMatch<'_>> {
        if max_skeleton_cost == 0 {
            return self.find_by_phonetic_skeleton(input, max_results);
        }
        if self.skeleton_index.is_empty() || max_results == 0 {
            return Vec::new();
        }

        let key = phonetic_skeleton(input);
        if key.is_empty() {
            return Vec::new();
        }

        let mut entry_indexes = Vec::new();
        let seed_limit = skeleton_seed_limit(max_results);
        collect_skeleton_key_matches(&self.skeleton_index, &key, &mut entry_indexes, seed_limit);
        collect_skeleton_key_matches(
            &self.skeleton_delete_index,
            &key,
            &mut entry_indexes,
            seed_limit,
        );

        for deletion in skeleton_deletions(&key) {
            collect_skeleton_key_matches(
                &self.skeleton_index,
                &deletion,
                &mut entry_indexes,
                seed_limit,
            );
            collect_skeleton_key_matches(
                &self.skeleton_delete_index,
                &deletion,
                &mut entry_indexes,
                seed_limit,
            );
        }

        entry_indexes.sort_unstable();
        entry_indexes.dedup();

        let mut matches = entry_indexes
            .into_iter()
            .filter_map(|entry_index| {
                let entry = &self.entries[entry_index];
                let entry_key = phonetic_skeleton(&entry.word);
                (skeleton_edit_distance(&key, &entry_key, max_skeleton_cost)? <= max_skeleton_cost)
                    .then(|| LexiconMatch {
                        entry,
                        edit_cost: weighted_edit_distance(input, &entry.word),
                    })
            })
            .collect::<Vec<_>>();

        matches.sort_by(|left, right| {
            left.edit_cost
                .cmp(&right.edit_cost)
                .then_with(|| right.entry.frequency.cmp(&left.entry.frequency))
                .then_with(|| left.entry.word.cmp(&right.entry.word))
        });
        matches.truncate(max_results);
        matches
    }

    fn find_within_edit_cost_at<'a>(
        &'a self,
        node_index: usize,
        unit: &str,
        input_units: &[&str],
        depth: usize,
        rows: &mut Vec<Vec<u16>>,
        max_edit_cost: u16,
        min_candidate_depth: usize,
        max_candidate_depth: usize,
        matches: &mut Vec<LexiconMatch<'a>>,
    ) {
        if depth > max_candidate_depth {
            return;
        }

        let node = &self.trie[node_index];
        if !subtree_length_can_match(node, depth, min_candidate_depth, max_candidate_depth) {
            return;
        }

        while rows.len() <= depth {
            rows.push(Vec::with_capacity(input_units.len() + 1));
        }

        let (final_cost, min_cost) = {
            let (previous_rows, current_rows) = rows.split_at_mut(depth);
            let previous_row = &previous_rows[depth - 1];
            let current_row = &mut current_rows[0];
            current_row.clear();
            current_row.push(previous_row[0] + INSERT_DELETE_COST);

            for (input_index, input_unit) in input_units.iter().enumerate() {
                let substitution = previous_row[input_index] + unit_similarity(input_unit, unit);
                let deletion = previous_row[input_index + 1] + INSERT_DELETE_COST;
                let insertion = current_row[input_index] + INSERT_DELETE_COST;
                current_row.push(substitution.min(deletion).min(insertion));
            }

            (
                current_row[input_units.len()],
                current_row.iter().copied().min().unwrap_or(u16::MAX),
            )
        };

        if final_cost <= max_edit_cost {
            if let Some(entry_index) = node.entry_index {
                matches.push(LexiconMatch {
                    entry: &self.entries[entry_index],
                    edit_cost: EditCost(final_cost),
                });
            }
        }

        if min_cost > max_edit_cost {
            return;
        }

        for (next_unit, child_index) in &node.children {
            self.find_within_edit_cost_at(
                *child_index,
                next_unit,
                input_units,
                depth + 1,
                rows,
                max_edit_cost,
                min_candidate_depth,
                max_candidate_depth,
                matches,
            );
        }
    }

    fn from_sorted_entries(entries: Vec<LexiconEntry>) -> Self {
        Self {
            trie: build_trie(&entries),
            skeleton_index: build_skeleton_index(&entries),
            skeleton_delete_index: build_skeleton_delete_index(&entries),
            entries,
        }
    }
}

impl LexiconEntry {
    pub fn new(word: impl Into<String>, frequency: u32) -> Self {
        Self {
            word: word.into(),
            frequency,
        }
    }
}

fn build_trie(entries: &[LexiconEntry]) -> Vec<LexiconNode> {
    let mut trie = vec![LexiconNode::default()];
    for (entry_index, entry) in entries.iter().enumerate() {
        insert_trie_entry(&mut trie, entry_index, &entry.word);
    }
    annotate_terminal_depths(&mut trie, 0);
    trie
}

fn build_skeleton_index(entries: &[LexiconEntry]) -> Vec<SkeletonIndexEntry> {
    let mut index = entries
        .iter()
        .enumerate()
        .filter_map(|(entry_index, entry)| {
            let key = phonetic_skeleton(&entry.word);
            (!key.is_empty()).then_some(SkeletonIndexEntry {
                key_hash: stable_hash(&key),
                entry_index,
            })
        })
        .collect::<Vec<_>>();

    index.sort_by(|left, right| {
        left.key_hash
            .cmp(&right.key_hash)
            .then_with(|| {
                entries[right.entry_index]
                    .frequency
                    .cmp(&entries[left.entry_index].frequency)
            })
            .then_with(|| {
                entries[left.entry_index]
                    .word
                    .cmp(&entries[right.entry_index].word)
            })
    });
    index
}

fn build_skeleton_delete_index(entries: &[LexiconEntry]) -> Vec<SkeletonIndexEntry> {
    let mut index = Vec::new();

    for (entry_index, entry) in entries.iter().enumerate() {
        let key = phonetic_skeleton(&entry.word);
        for deletion in skeleton_deletions(&key) {
            index.push(SkeletonIndexEntry {
                key_hash: stable_hash(&deletion),
                entry_index,
            });
        }
    }

    index.sort_by(|left, right| {
        left.key_hash
            .cmp(&right.key_hash)
            .then_with(|| {
                entries[right.entry_index]
                    .frequency
                    .cmp(&entries[left.entry_index].frequency)
            })
            .then_with(|| {
                entries[left.entry_index]
                    .word
                    .cmp(&entries[right.entry_index].word)
            })
    });
    index
}

fn collect_skeleton_key_matches(
    index: &[SkeletonIndexEntry],
    key: &str,
    entry_indexes: &mut Vec<usize>,
    seed_limit: usize,
) {
    if entry_indexes.len() >= seed_limit {
        return;
    }

    let key_hash = stable_hash(key);
    let start = index.partition_point(|entry| entry.key_hash < key_hash);
    let remaining = seed_limit - entry_indexes.len();
    entry_indexes.extend(
        index[start..]
            .iter()
            .take_while(|entry| entry.key_hash == key_hash)
            .take(remaining)
            .map(|entry| entry.entry_index),
    );
}

fn skeleton_seed_limit(max_results: usize) -> usize {
    max_results.saturating_mul(64).max(max_results).max(64)
}

fn stable_hash(text: &str) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    text.as_bytes().iter().fold(OFFSET, |hash, byte| {
        hash.wrapping_mul(PRIME) ^ u64::from(*byte)
    })
}

fn skeleton_deletions(key: &str) -> Vec<String> {
    let chars = key.chars().collect::<Vec<_>>();
    if chars.len() <= 1 {
        return Vec::new();
    }

    let mut deletions = Vec::with_capacity(chars.len());
    for delete_index in 0..chars.len() {
        let mut deletion = String::with_capacity(key.len());
        for (index, ch) in chars.iter().enumerate() {
            if index != delete_index {
                deletion.push(*ch);
            }
        }
        if !deletions.contains(&deletion) {
            deletions.push(deletion);
        }
    }
    deletions
}

fn skeleton_edit_distance(left: &str, right: &str, max_cost: u16) -> Option<u16> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > max_cost as usize {
        return None;
    }

    let mut previous = (0..=right.len())
        .map(|index| index as u16)
        .collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];

    for (left_index, left_ch) in left.iter().enumerate() {
        current[0] = (left_index + 1) as u16;
        let mut row_min = current[0];
        for (right_index, right_ch) in right.iter().enumerate() {
            let substitution = previous[right_index] + u16::from(left_ch != right_ch);
            let deletion = previous[right_index + 1] + 1;
            let insertion = current[right_index] + 1;
            let cost = substitution.min(deletion).min(insertion);
            current[right_index + 1] = cost;
            row_min = row_min.min(cost);
        }
        if row_min > max_cost {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }

    (previous[right.len()] <= max_cost).then_some(previous[right.len()])
}

fn insert_trie_entry(trie: &mut Vec<LexiconNode>, entry_index: usize, word: &str) {
    let mut node_index = 0;
    for unit in bangla_units(word) {
        let search = trie[node_index]
            .children
            .binary_search_by(|(child, _)| child.as_str().cmp(unit));
        let child_index = match search {
            Ok(index) => trie[node_index].children[index].1,
            Err(index) => {
                let new_index = trie.len();
                trie.push(LexiconNode::default());
                trie[node_index]
                    .children
                    .insert(index, (unit.to_string(), new_index));
                new_index
            }
        };
        node_index = child_index;
    }
    trie[node_index].entry_index = Some(entry_index);
}

fn annotate_terminal_depths(trie: &mut [LexiconNode], node_index: usize) -> (u16, u16) {
    let mut min_depth = if trie[node_index].entry_index.is_some() {
        0
    } else {
        u16::MAX
    };
    let mut max_depth = 0;
    let children = trie[node_index]
        .children
        .iter()
        .map(|(_, child_index)| *child_index)
        .collect::<Vec<_>>();

    for child_index in children {
        let (child_min, child_max) = annotate_terminal_depths(trie, child_index);
        min_depth = min_depth.min(child_min.saturating_add(1));
        max_depth = max_depth.max(child_max.saturating_add(1));
    }

    if min_depth == u16::MAX {
        min_depth = 0;
    }

    trie[node_index].min_terminal_depth = min_depth;
    trie[node_index].max_terminal_depth = max_depth;
    (min_depth, max_depth)
}

fn subtree_length_can_match(
    node: &LexiconNode,
    depth: usize,
    min_candidate_depth: usize,
    max_candidate_depth: usize,
) -> bool {
    let subtree_min = depth + usize::from(node.min_terminal_depth);
    let subtree_max = depth + usize::from(node.max_terminal_depth);
    subtree_min <= max_candidate_depth && subtree_max >= min_candidate_depth
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Lexicon, LexiconArtifactError, LexiconEntry};
    use crate::autocorrect::weighted_edit_distance;

    #[test]
    fn lexicon_deduplicates_by_highest_frequency() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আমি", 20),
            LexiconEntry::new("", 100),
        ]);

        assert_eq!(lexicon.len(), 1);
        assert_eq!(lexicon.frequency("আমি"), Some(20));
    }

    #[test]
    fn lexicon_stats_describe_runtime_index_shape() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আমার", 9),
            LexiconEntry::new("বিজ্ঞান", 8),
        ]);
        let stats = lexicon.stats();

        assert_eq!(stats.entries, 3);
        assert!(stats.trie_nodes >= stats.entries);
        assert!(stats.trie_edges >= stats.entries);
        assert_eq!(stats.skeleton_keys, 3);
        assert!(stats.skeleton_delete_keys >= stats.entries);
    }

    #[test]
    fn skeleton_index_entries_remain_compact() {
        assert!(std::mem::size_of::<super::SkeletonIndexEntry>() <= 16);
    }

    #[test]
    fn trie_lookup_finds_bangla_unit_edit_candidates() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আমার", 9),
            LexiconEntry::new("বিজ্ঞান", 8),
        ]);

        let matches = lexicon.find_within_edit_cost("আমী", 2);
        let words = matches
            .iter()
            .map(|candidate| candidate.entry.word.as_str())
            .collect::<Vec<_>>();

        assert!(words.contains(&"আমি"));
        assert!(!words.contains(&"বিজ্ঞান"));
    }

    #[test]
    fn skeleton_lookup_finds_vowel_variant_candidates() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("কিরণ", 10),
            LexiconEntry::new("করণ", 100),
            LexiconEntry::new("বিজ্ঞান", 8),
        ]);

        let matches = lexicon.find_by_phonetic_skeleton("কীরন", 8);
        let words = matches
            .iter()
            .map(|candidate| candidate.entry.word.as_str())
            .collect::<Vec<_>>();

        assert_eq!(words, vec!["করণ", "কিরণ"]);
    }

    #[test]
    fn skeleton_lookup_is_bounded() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("কিরণ", 10),
            LexiconEntry::new("করণ", 100),
            LexiconEntry::new("কোরণ", 50),
        ]);

        let matches = lexicon.find_by_phonetic_skeleton("কীরন", 1);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.word, "করণ");
    }

    #[test]
    fn fuzzy_skeleton_lookup_finds_one_edit_skeleton_variants() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("কিরণ", 10),
            LexiconEntry::new("কারণ", 100),
            LexiconEntry::new("বরফ", 50),
            LexiconEntry::new("বিজ্ঞান", 8),
        ]);

        let insertion_matches = lexicon.find_by_fuzzy_phonetic_skeleton("করন", 1, 8);
        let insertion_words = insertion_matches
            .iter()
            .map(|candidate| candidate.entry.word.as_str())
            .collect::<Vec<_>>();
        assert!(insertion_words.contains(&"কিরণ"));
        assert!(insertion_words.contains(&"কারণ"));

        let substitution_matches = lexicon.find_by_fuzzy_phonetic_skeleton("বরব", 1, 8);
        let substitution_words = substitution_matches
            .iter()
            .map(|candidate| candidate.entry.word.as_str())
            .collect::<Vec<_>>();
        assert!(substitution_words.contains(&"বরফ"));
    }

    #[test]
    fn trie_lookup_costs_match_standalone_weighted_edit_distance() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আমার", 9),
            LexiconEntry::new("আমরা", 8),
            LexiconEntry::new("বিজ্ঞান", 7),
            LexiconEntry::new("কিরণ", 6),
            LexiconEntry::new("করণ", 5),
        ]);

        for input in ["আমী", "আমাদের", "বিজান", "কীরণ"] {
            let max_edit_cost = 4;
            let actual = lexicon
                .find_within_edit_cost(input, max_edit_cost)
                .into_iter()
                .map(|candidate| (candidate.entry.word.as_str(), candidate.edit_cost))
                .collect::<BTreeMap<_, _>>();
            let expected = lexicon
                .entries()
                .iter()
                .filter_map(|entry| {
                    let cost = weighted_edit_distance(input, &entry.word);
                    (cost.0 <= max_edit_cost).then_some((entry.word.as_str(), cost))
                })
                .collect::<BTreeMap<_, _>>();

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn compact_artifact_round_trips_lexicon_entries() {
        let lexicon = Lexicon::new([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আমার", 9),
            LexiconEntry::new("বিজ্ঞান", 8),
        ]);

        let bytes = lexicon.to_compact_bytes().expect("artifact should encode");
        let decoded = Lexicon::from_compact_bytes(&bytes).expect("artifact should decode");

        assert_eq!(decoded.entries(), lexicon.entries());
        assert_eq!(decoded.frequency("বিজ্ঞান"), Some(8));
    }

    #[test]
    fn compact_artifact_rejects_corrupt_inputs() {
        assert_eq!(
            Lexicon::from_compact_bytes(b"bad").unwrap_err(),
            LexiconArtifactError::Truncated
        );

        let lexicon = Lexicon::new([LexiconEntry::new("আমি", 10)]);
        let mut bytes = lexicon.to_compact_bytes().expect("artifact should encode");
        bytes[0] = b'X';
        assert_eq!(
            Lexicon::from_compact_bytes(&bytes).unwrap_err(),
            LexiconArtifactError::InvalidMagic
        );

        let mut duplicate = Vec::new();
        duplicate.extend_from_slice(super::LEXICON_MAGIC);
        super::push_u32(&mut duplicate, 2);
        for _ in 0..2 {
            super::push_u32(&mut duplicate, 1);
            super::push_u16(&mut duplicate, "আমি".len() as u16);
            duplicate.extend_from_slice("আমি".as_bytes());
        }
        assert_eq!(
            Lexicon::from_compact_bytes(&duplicate).unwrap_err(),
            LexiconArtifactError::DuplicateWord { index: 1 }
        );
    }
}
