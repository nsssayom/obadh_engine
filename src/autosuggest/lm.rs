use serde::Serialize;

use super::artifact::{
    parse_layout, read_i32, read_u32, token_slice, AutosuggestArtifactError, Layout,
    BIGRAM_ROW_LEN, CANDIDATE_RECORD_LEN, ID_TOKEN_RECORD_LEN, TOKEN_INDEX_RECORD_LEN,
    TRIGRAM_ROW_LEN,
};

pub const DEFAULT_AUTOSUGGEST_CANDIDATES: usize = 5;
const PAD_ID: u32 = 0;
const BOS_ID: u32 = 1;
const UNK_ID: u32 = 2;

#[derive(Debug, Clone, Copy)]
pub struct AutosuggestOptions {
    pub max_candidates: usize,
}

impl Default for AutosuggestOptions {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutosuggestSource {
    Trigram,
    Bigram,
    Unigram,
}

impl AutosuggestSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trigram => "trigram",
            Self::Bigram => "bigram",
            Self::Unigram => "unigram",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestCandidate<'a> {
    pub text: &'a str,
    pub token_id: u32,
    pub source: AutosuggestSource,
    pub count: u32,
    pub score: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestResult<'a> {
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
    pub candidates: Vec<AutosuggestCandidate<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestMetadata {
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
}

#[derive(Debug, Clone)]
pub struct AutosuggestLm<D: AsRef<[u8]> = Vec<u8>> {
    bytes: D,
    layout: Layout,
}

impl<D: AsRef<[u8]>> AutosuggestLm<D> {
    pub fn from_bytes(bytes: D) -> Result<Self, AutosuggestArtifactError> {
        let layout = parse_layout(bytes.as_ref())?;
        Ok(Self { bytes, layout })
    }

    pub fn vocab_size(&self) -> usize {
        self.layout.header.vocab_size as usize
    }

    pub fn unigram_count(&self) -> usize {
        self.layout.header.unigram_count as usize
    }

    pub fn bigram_row_count(&self) -> usize {
        self.layout.header.bigram_row_count as usize
    }

    pub fn trigram_row_count(&self) -> usize {
        self.layout.header.trigram_row_count as usize
    }

    pub fn artifact_bytes(&self) -> usize {
        self.layout.sections.end
    }

    pub fn token_id(&self, token: &str) -> Result<Option<u32>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let target = token.as_bytes();
        let mut low = 0_usize;
        let mut high = self.layout.header.token_index_count as usize;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.token_index + mid * TOKEN_INDEX_RECORD_LEN;
            let token_offset = read_u32(bytes, offset)?;
            let token_len = read_u32(bytes, offset + 4)?;
            let id = read_u32(bytes, offset + 8)?;
            let token = self.token_bytes(token_offset, token_len)?;
            match token.cmp(target) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(Some(id)),
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    pub fn token_text(&self, token_id: u32) -> Result<&str, AutosuggestArtifactError> {
        if token_id >= self.layout.header.vocab_size {
            return Err(AutosuggestArtifactError::InvalidTokenId(token_id));
        }
        let offset = self.layout.sections.id_tokens + token_id as usize * ID_TOKEN_RECORD_LEN;
        let token_offset = read_u32(self.bytes.as_ref(), offset)?;
        let token_len = read_u32(self.bytes.as_ref(), offset + 4)?;
        token_slice(self.bytes.as_ref(), self.layout, token_offset, token_len)
    }

    pub fn suggest_for_text(
        &self,
        context: &str,
        options: AutosuggestOptions,
    ) -> Result<AutosuggestResult<'_>, AutosuggestArtifactError> {
        let mut candidates = Vec::with_capacity(options.max_candidates.max(1));
        let metadata = self.suggest_for_text_into(context, options, &mut candidates)?;
        Ok(AutosuggestResult {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            candidates,
        })
    }

    pub fn suggest_for_text_into<'a>(
        &'a self,
        context: &str,
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let mut token_count = 0_usize;
        let mut ids = [0_u32; 2];
        let mut id_len = 0_usize;
        output.clear();

        for raw_token in context.split_whitespace() {
            let Some(token) = clean_context_token(raw_token) else {
                continue;
            };
            token_count += 1;
            match self.token_id(token)? {
                Some(id) if id > UNK_ID => {
                    push_recent_context_id(&mut ids, &mut id_len, id);
                }
                Some(BOS_ID) | Some(PAD_ID) | Some(UNK_ID) | None => id_len = 0,
                Some(_) => id_len = 0,
            }
        }

        self.suggest_for_token_ids_into(token_count, &ids[..id_len], options, output)
    }

    pub fn suggest_for_tokens(
        &self,
        context_tokens: &[&str],
        options: AutosuggestOptions,
    ) -> Result<AutosuggestResult<'_>, AutosuggestArtifactError> {
        let mut candidates = Vec::with_capacity(options.max_candidates.max(1));
        let metadata = self.suggest_for_tokens_into(context_tokens, options, &mut candidates)?;
        Ok(AutosuggestResult {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            candidates,
        })
    }

    pub fn suggest_for_tokens_into<'a>(
        &'a self,
        context_tokens: &[&str],
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let mut ids = [0_u32; 2];
        let mut id_len = 0_usize;
        output.clear();

        for token in context_tokens {
            match self.token_id(token)? {
                Some(id) if id > UNK_ID => {
                    push_recent_context_id(&mut ids, &mut id_len, id);
                }
                Some(BOS_ID) | Some(PAD_ID) | Some(UNK_ID) | None => id_len = 0,
                Some(_) => id_len = 0,
            }
        }
        self.suggest_for_token_ids_into(context_tokens.len(), &ids[..id_len], options, output)
    }

    fn suggest_for_token_ids_into<'a>(
        &'a self,
        context_token_count: usize,
        context_ids: &[u32],
        options: AutosuggestOptions,
        candidates: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let limit = options.max_candidates.max(1);

        if let [prefix1, prefix2] = context_ids {
            if let Some(row) = self.find_trigram_row(*prefix1, *prefix2)? {
                self.collect_candidates(
                    row.0,
                    row.1,
                    AutosuggestSource::Trigram,
                    limit,
                    candidates,
                )?;
            }
        }

        if candidates.len() < limit {
            if let Some(prefix) = context_ids.last().copied() {
                if let Some(row) = self.find_bigram_row(prefix)? {
                    self.collect_candidates(
                        row.0,
                        row.1,
                        AutosuggestSource::Bigram,
                        limit,
                        candidates,
                    )?;
                }
            }
        }

        if candidates.len() < limit {
            self.collect_unigrams(limit, candidates)?;
        }

        Ok(AutosuggestMetadata {
            context_token_count,
            matched_context_token_count: context_ids.len(),
        })
    }

    fn find_bigram_row(&self, prefix: u32) -> Result<Option<(u32, u32)>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let mut low = 0_usize;
        let mut high = self.layout.header.bigram_row_count as usize;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.bigram_rows + mid * BIGRAM_ROW_LEN;
            let row_prefix = read_u32(bytes, offset)?;
            match row_prefix.cmp(&prefix) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => {
                    let start = read_u32(bytes, offset + 4)?;
                    let len = read_u32(bytes, offset + 8)?;
                    return Ok(Some((start, len)));
                }
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    fn find_trigram_row(
        &self,
        prefix1: u32,
        prefix2: u32,
    ) -> Result<Option<(u32, u32)>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let mut low = 0_usize;
        let mut high = self.layout.header.trigram_row_count as usize;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.trigram_rows + mid * TRIGRAM_ROW_LEN;
            let row_prefix1 = read_u32(bytes, offset)?;
            let row_prefix2 = read_u32(bytes, offset + 4)?;
            match (row_prefix1, row_prefix2).cmp(&(prefix1, prefix2)) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => {
                    let start = read_u32(bytes, offset + 8)?;
                    let len = read_u32(bytes, offset + 12)?;
                    return Ok(Some((start, len)));
                }
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    fn collect_unigrams<'a>(
        &'a self,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let len = self.layout.header.unigram_count as usize;
        for index in 0..len {
            if output.len() >= limit {
                break;
            }
            let offset = self.layout.sections.unigrams + index * CANDIDATE_RECORD_LEN;
            self.collect_candidate_at(offset, AutosuggestSource::Unigram, output)?;
        }
        Ok(())
    }

    fn collect_candidates<'a>(
        &'a self,
        start: u32,
        len: u32,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let start = start as usize;
        let len = len as usize;
        let end = start
            .checked_add(len)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        if end > self.layout.header.candidate_count as usize {
            return Err(AutosuggestArtifactError::InvalidSectionLayout);
        }

        for index in start..end {
            if output.len() >= limit {
                break;
            }
            let offset = self.layout.sections.candidates + index * CANDIDATE_RECORD_LEN;
            self.collect_candidate_at(offset, source, output)?;
        }

        Ok(())
    }

    fn collect_candidate_at<'a>(
        &'a self,
        offset: usize,
        source: AutosuggestSource,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let token_id = read_u32(bytes, offset)?;
        if token_id <= UNK_ID
            || output
                .iter()
                .any(|candidate| candidate.token_id == token_id)
        {
            return Ok(());
        }
        let count = read_u32(bytes, offset + 4)?;
        let score = read_i32(bytes, offset + 8)?;
        let text = self.token_text(token_id)?;
        output.push(AutosuggestCandidate {
            text,
            token_id,
            source,
            count,
            score,
        });
        Ok(())
    }

    fn token_bytes(&self, offset: u32, len: u32) -> Result<&[u8], AutosuggestArtifactError> {
        let start = self
            .layout
            .sections
            .token_bytes
            .checked_add(offset as usize)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        let end = start
            .checked_add(len as usize)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        self.bytes
            .as_ref()
            .get(start..end)
            .ok_or(AutosuggestArtifactError::UnexpectedEof)
    }
}

fn push_recent_context_id(ids: &mut [u32; 2], len: &mut usize, id: u32) {
    if *len < 2 {
        ids[*len] = id;
        *len += 1;
    } else {
        ids[0] = ids[1];
        ids[1] = id;
    }
}

fn clean_context_token(raw: &str) -> Option<&str> {
    let token = raw.trim_matches(|ch: char| {
        ch.is_ascii_punctuation()
            || matches!(
                ch,
                '।' | '॥' | ',' | ';' | ':' | '!' | '?' | '"' | '\'' | '(' | ')' | '[' | ']'
            )
    });
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autosuggest::artifact::test_support::{build_fixture, Row};

    fn fixture() -> AutosuggestLm<Vec<u8>> {
        let tokens = [
            "<pad>",
            "<bos>",
            "<unk>",
            "আমি",
            "আজ",
            "সকালে",
            "স্কুলে",
            "যাব",
            "খাব",
        ];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 100), (7, 90), (8, 80)],
            &[
                Row {
                    context: vec![4],
                    candidates: vec![(5, 40), (7, 12)],
                },
                Row {
                    context: vec![3, 4],
                    candidates: vec![(7, 9), (5, 7)],
                },
            ],
        ))
        .expect("fixture should parse")
    }

    #[test]
    fn token_lookup_uses_sorted_index() {
        let lm = fixture();
        assert_eq!(lm.token_id("আজ").unwrap(), Some(4));
        assert_eq!(lm.token_id("নেই").unwrap(), None);
        assert_eq!(lm.token_text(7).unwrap(), "যাব");
    }

    #[test]
    fn trigram_candidates_backoff_without_duplicates() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি আজ", AutosuggestOptions { max_candidates: 4 })
            .unwrap();
        let rows = result
            .candidates
            .iter()
            .map(|candidate| (candidate.text, candidate.source))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![
                ("যাব", AutosuggestSource::Trigram),
                ("সকালে", AutosuggestSource::Trigram),
                ("খাব", AutosuggestSource::Unigram)
            ]
        );
        assert_eq!(result.matched_context_token_count, 2);
    }

    #[test]
    fn unknown_context_falls_back_to_unigram() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি অচেনা", AutosuggestOptions { max_candidates: 2 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
        assert_eq!(result.matched_context_token_count, 0);
    }

    #[test]
    fn into_api_reuses_caller_candidate_buffer() {
        let lm = fixture();
        let mut candidates = Vec::with_capacity(4);
        let initial_capacity = candidates.capacity();

        let metadata = lm
            .suggest_for_text_into(
                "আমি আজ",
                AutosuggestOptions { max_candidates: 4 },
                &mut candidates,
            )
            .unwrap();
        assert_eq!(metadata.matched_context_token_count, 2);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates.capacity(), initial_capacity);

        let metadata = lm
            .suggest_for_text_into(
                "আমি অচেনা",
                AutosuggestOptions { max_candidates: 2 },
                &mut candidates,
            )
            .unwrap();
        assert_eq!(metadata.matched_context_token_count, 0);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
        assert_eq!(candidates.capacity(), initial_capacity);
    }
}
