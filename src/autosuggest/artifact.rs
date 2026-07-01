use std::error::Error;
use std::fmt;
use std::str;

pub(crate) const MAGIC: &[u8; 16] = b"OBAUTOSUGLM_V1\0\0";
pub(crate) const MAGIC_V2: &[u8; 16] = b"OBAUTOSUGLM_V2\0\0";
pub(crate) const MAGIC_V3: &[u8; 16] = b"OBAUTOSUGLM_V3\0\0";
pub(crate) const VERSION: u32 = 1;
pub(crate) const VERSION_V2: u32 = 2;
pub(crate) const VERSION_V3: u32 = 3;

pub(crate) const HEADER_LEN: usize = 52;
pub(crate) const HEADER_LEN_V2: usize = 56;
pub(crate) const HEADER_LEN_V3: usize = 60;
pub(crate) const ID_TOKEN_RECORD_LEN: usize = 8;
pub(crate) const TOKEN_INDEX_RECORD_LEN: usize = 12;
pub(crate) const CANDIDATE_RECORD_LEN: usize = 12;
pub(crate) const COUNT_CANDIDATE_RECORD_LEN: usize = 8;
pub(crate) const BIGRAM_ROW_LEN: usize = 12;
pub(crate) const TRIGRAM_ROW_LEN: usize = 16;
pub(crate) const FOURGRAM_ROW_LEN: usize = 20;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Header {
    pub(crate) version: u32,
    pub(crate) vocab_size: u32,
    pub(crate) token_index_count: u32,
    pub(crate) unigram_count: u32,
    pub(crate) bigram_row_count: u32,
    pub(crate) trigram_row_count: u32,
    pub(crate) fourgram_row_count: u32,
    pub(crate) candidate_count: u32,
    pub(crate) candidate_record_len: u32,
    pub(crate) token_bytes_len: u32,
    pub(crate) vocab_fingerprint: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Sections {
    pub(crate) id_tokens: usize,
    pub(crate) token_index: usize,
    pub(crate) unigrams: usize,
    pub(crate) bigram_rows: usize,
    pub(crate) trigram_rows: usize,
    pub(crate) fourgram_rows: usize,
    pub(crate) candidates: usize,
    pub(crate) token_bytes: usize,
    pub(crate) end: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) header: Header,
    pub(crate) sections: Sections,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutosuggestArtifactError {
    UnexpectedEof,
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidSectionLayout,
    InvalidTokenId(u32),
    ModelFingerprintMismatch { expected: u32, actual: u32 },
    InvalidUtf8,
}

impl fmt::Display for AutosuggestArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("autosuggest artifact is truncated"),
            Self::InvalidMagic => f.write_str("autosuggest artifact has an invalid magic header"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported autosuggest artifact version {version}")
            }
            Self::InvalidSectionLayout => {
                f.write_str("autosuggest artifact section layout is invalid")
            }
            Self::InvalidTokenId(id) => {
                write!(f, "autosuggest artifact references invalid token id {id}")
            }
            Self::ModelFingerprintMismatch { expected, actual } => {
                write!(
                    f,
                    "autosuggest model fingerprint {actual:08x} does not match expected fingerprint {expected:08x}"
                )
            }
            Self::InvalidUtf8 => {
                f.write_str("autosuggest artifact contains invalid UTF-8 token bytes")
            }
        }
    }
}

impl Error for AutosuggestArtifactError {}

pub(crate) fn parse_layout(bytes: &[u8]) -> Result<Layout, AutosuggestArtifactError> {
    if bytes.len() < HEADER_LEN {
        return Err(AutosuggestArtifactError::UnexpectedEof);
    }
    let magic = bytes
        .get(..MAGIC.len())
        .ok_or(AutosuggestArtifactError::UnexpectedEof)?;
    if magic != MAGIC && magic != MAGIC_V2 && magic != MAGIC_V3 {
        return Err(AutosuggestArtifactError::InvalidMagic);
    }

    let version = read_u32(bytes, 16)?;
    if (magic == MAGIC && version != VERSION)
        || (magic == MAGIC_V2 && version != VERSION_V2)
        || (magic == MAGIC_V3 && version != VERSION_V3)
    {
        return Err(AutosuggestArtifactError::UnsupportedVersion(version));
    }
    let header_len = match version {
        VERSION => HEADER_LEN,
        VERSION_V2 => {
            if bytes.len() < HEADER_LEN_V2 {
                return Err(AutosuggestArtifactError::UnexpectedEof);
            }
            HEADER_LEN_V2
        }
        VERSION_V3 => {
            if bytes.len() < HEADER_LEN_V3 {
                return Err(AutosuggestArtifactError::UnexpectedEof);
            }
            HEADER_LEN_V3
        }
        _ => return Err(AutosuggestArtifactError::UnsupportedVersion(version)),
    };

    let header = match version {
        VERSION => Header {
            version,
            vocab_size: read_u32(bytes, 20)?,
            token_index_count: read_u32(bytes, 24)?,
            unigram_count: read_u32(bytes, 28)?,
            bigram_row_count: read_u32(bytes, 32)?,
            trigram_row_count: read_u32(bytes, 36)?,
            fourgram_row_count: 0,
            candidate_count: read_u32(bytes, 40)?,
            candidate_record_len: CANDIDATE_RECORD_LEN as u32,
            token_bytes_len: read_u32(bytes, 44)?,
            vocab_fingerprint: read_u32(bytes, 48)?,
        },
        VERSION_V2 => Header {
            version,
            vocab_size: read_u32(bytes, 20)?,
            token_index_count: read_u32(bytes, 24)?,
            unigram_count: read_u32(bytes, 28)?,
            bigram_row_count: read_u32(bytes, 32)?,
            trigram_row_count: read_u32(bytes, 36)?,
            fourgram_row_count: read_u32(bytes, 40)?,
            candidate_count: read_u32(bytes, 44)?,
            candidate_record_len: CANDIDATE_RECORD_LEN as u32,
            token_bytes_len: read_u32(bytes, 48)?,
            vocab_fingerprint: read_u32(bytes, 52)?,
        },
        VERSION_V3 => Header {
            version,
            vocab_size: read_u32(bytes, 20)?,
            token_index_count: read_u32(bytes, 24)?,
            unigram_count: read_u32(bytes, 28)?,
            bigram_row_count: read_u32(bytes, 32)?,
            trigram_row_count: read_u32(bytes, 36)?,
            fourgram_row_count: read_u32(bytes, 40)?,
            candidate_count: read_u32(bytes, 44)?,
            token_bytes_len: read_u32(bytes, 48)?,
            vocab_fingerprint: read_u32(bytes, 52)?,
            candidate_record_len: read_u32(bytes, 56)?,
        },
        _ => return Err(AutosuggestArtifactError::UnsupportedVersion(version)),
    };

    if header.vocab_size == 0 || header.token_index_count != header.vocab_size {
        return Err(AutosuggestArtifactError::InvalidSectionLayout);
    }
    if header.candidate_record_len as usize != CANDIDATE_RECORD_LEN
        && header.candidate_record_len as usize != COUNT_CANDIDATE_RECORD_LEN
    {
        return Err(AutosuggestArtifactError::InvalidSectionLayout);
    }

    let id_tokens = header_len;
    let token_index = checked_advance(
        id_tokens,
        header.vocab_size as usize,
        ID_TOKEN_RECORD_LEN,
        bytes.len(),
    )?;
    let unigrams = checked_advance(
        token_index,
        header.token_index_count as usize,
        TOKEN_INDEX_RECORD_LEN,
        bytes.len(),
    )?;
    let bigram_rows = checked_advance(
        unigrams,
        header.unigram_count as usize,
        header.candidate_record_len as usize,
        bytes.len(),
    )?;
    let trigram_rows = checked_advance(
        bigram_rows,
        header.bigram_row_count as usize,
        BIGRAM_ROW_LEN,
        bytes.len(),
    )?;
    let fourgram_rows = checked_advance(
        trigram_rows,
        header.trigram_row_count as usize,
        TRIGRAM_ROW_LEN,
        bytes.len(),
    )?;
    let candidates = checked_advance(
        fourgram_rows,
        header.fourgram_row_count as usize,
        FOURGRAM_ROW_LEN,
        bytes.len(),
    )?;
    let token_bytes = checked_advance(
        candidates,
        header.candidate_count as usize,
        header.candidate_record_len as usize,
        bytes.len(),
    )?;
    let end = checked_advance(token_bytes, header.token_bytes_len as usize, 1, bytes.len())?;

    if end != bytes.len() {
        return Err(AutosuggestArtifactError::InvalidSectionLayout);
    }

    Ok(Layout {
        header,
        sections: Sections {
            id_tokens,
            token_index,
            unigrams,
            bigram_rows,
            trigram_rows,
            fourgram_rows,
            candidates,
            token_bytes,
            end,
        },
    })
}

pub(crate) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, AutosuggestArtifactError> {
    let end = offset
        .checked_add(4)
        .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(AutosuggestArtifactError::UnexpectedEof)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub(crate) fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, AutosuggestArtifactError> {
    let end = offset
        .checked_add(4)
        .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(AutosuggestArtifactError::UnexpectedEof)?;
    Ok(i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub(crate) fn token_slice<'a>(
    bytes: &'a [u8],
    layout: Layout,
    offset: u32,
    len: u32,
) -> Result<&'a str, AutosuggestArtifactError> {
    let start = (layout.sections.token_bytes)
        .checked_add(offset as usize)
        .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
    let end = start
        .checked_add(len as usize)
        .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
    let slice = bytes
        .get(start..end)
        .ok_or(AutosuggestArtifactError::UnexpectedEof)?;
    str::from_utf8(slice).map_err(|_| AutosuggestArtifactError::InvalidUtf8)
}

fn checked_advance(
    offset: usize,
    count: usize,
    record_len: usize,
    total_len: usize,
) -> Result<usize, AutosuggestArtifactError> {
    let bytes = count
        .checked_mul(record_len)
        .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
    let end = offset
        .checked_add(bytes)
        .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
    if end > total_len {
        return Err(AutosuggestArtifactError::UnexpectedEof);
    }
    Ok(end)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{
        CANDIDATE_RECORD_LEN, COUNT_CANDIDATE_RECORD_LEN, MAGIC, MAGIC_V2, MAGIC_V3, VERSION,
        VERSION_V2, VERSION_V3,
    };

    #[derive(Debug, Clone)]
    pub(crate) struct Row {
        pub(crate) context: Vec<u32>,
        pub(crate) candidates: Vec<(u32, u32, i32)>,
    }

    pub(crate) fn build_fixture(
        tokens: &[&str],
        unigrams: &[(u32, u32, i32)],
        rows: &[Row],
    ) -> Vec<u8> {
        build_fixture_inner(tokens, unigrams, rows, false)
    }

    pub(crate) fn build_count_fixture(
        tokens: &[&str],
        unigrams: &[(u32, u32, i32)],
        rows: &[Row],
    ) -> Vec<u8> {
        build_fixture_inner(tokens, unigrams, rows, true)
    }

    fn build_fixture_inner(
        tokens: &[&str],
        unigrams: &[(u32, u32, i32)],
        rows: &[Row],
        compact_count_records: bool,
    ) -> Vec<u8> {
        let mut token_bytes = Vec::new();
        let mut id_records = Vec::new();
        for token in tokens {
            let offset = token_bytes.len() as u32;
            let bytes = token.as_bytes();
            token_bytes.extend_from_slice(bytes);
            id_records.push((offset, bytes.len() as u32));
        }

        let mut token_index = id_records
            .iter()
            .enumerate()
            .map(|(id, (offset, len))| {
                let start = *offset as usize;
                let end = start + *len as usize;
                (token_bytes[start..end].to_vec(), *offset, *len, id as u32)
            })
            .collect::<Vec<_>>();
        token_index.sort_by(|left, right| left.0.cmp(&right.0));

        let mut bigram_rows = Vec::new();
        let mut trigram_rows = Vec::new();
        let mut fourgram_rows = Vec::new();
        let mut candidates = Vec::new();

        for row in rows {
            let start = candidates.len() as u32;
            for (token_id, count, score) in &row.candidates {
                candidates.push((*token_id, *count, *score));
            }
            match row.context.as_slice() {
                [prefix] => bigram_rows.push((*prefix, start, row.candidates.len() as u32)),
                [prefix1, prefix2] => {
                    trigram_rows.push((*prefix1, *prefix2, start, row.candidates.len() as u32))
                }
                [prefix1, prefix2, prefix3] => fourgram_rows.push((
                    *prefix1,
                    *prefix2,
                    *prefix3,
                    start,
                    row.candidates.len() as u32,
                )),
                _ => panic!("fixture rows must be bigram, trigram, or fourgram contexts"),
            }
        }
        bigram_rows.sort_by_key(|row| row.0);
        trigram_rows.sort_by_key(|row| (row.0, row.1));
        fourgram_rows.sort_by_key(|row| (row.0, row.1, row.2));

        let mut bytes = Vec::new();
        let version = if compact_count_records {
            VERSION_V3
        } else if !fourgram_rows.is_empty() {
            VERSION_V2
        } else {
            VERSION
        };
        bytes.extend_from_slice(match version {
            VERSION => MAGIC,
            VERSION_V2 => MAGIC_V2,
            VERSION_V3 => MAGIC_V3,
            _ => unreachable!(),
        });
        write_u32(&mut bytes, version);
        write_u32(&mut bytes, tokens.len() as u32);
        write_u32(&mut bytes, tokens.len() as u32);
        write_u32(&mut bytes, unigrams.len() as u32);
        write_u32(&mut bytes, bigram_rows.len() as u32);
        write_u32(&mut bytes, trigram_rows.len() as u32);
        if version >= VERSION_V2 {
            write_u32(&mut bytes, fourgram_rows.len() as u32);
        }
        write_u32(&mut bytes, candidates.len() as u32);
        write_u32(&mut bytes, token_bytes.len() as u32);
        write_u32(&mut bytes, 0);
        if version >= VERSION_V3 {
            write_u32(
                &mut bytes,
                if compact_count_records {
                    COUNT_CANDIDATE_RECORD_LEN
                } else {
                    CANDIDATE_RECORD_LEN
                } as u32,
            );
        }

        for (offset, len) in &id_records {
            write_u32(&mut bytes, *offset);
            write_u32(&mut bytes, *len);
        }
        for (_, offset, len, id) in &token_index {
            write_u32(&mut bytes, *offset);
            write_u32(&mut bytes, *len);
            write_u32(&mut bytes, *id);
        }
        for (token_id, count, score) in unigrams {
            write_u32(&mut bytes, *token_id);
            write_u32(&mut bytes, *count);
            if !compact_count_records {
                write_i32(&mut bytes, *score);
            }
        }
        for (prefix, start, len) in &bigram_rows {
            write_u32(&mut bytes, *prefix);
            write_u32(&mut bytes, *start);
            write_u32(&mut bytes, *len);
        }
        for (prefix1, prefix2, start, len) in &trigram_rows {
            write_u32(&mut bytes, *prefix1);
            write_u32(&mut bytes, *prefix2);
            write_u32(&mut bytes, *start);
            write_u32(&mut bytes, *len);
        }
        for (prefix1, prefix2, prefix3, start, len) in &fourgram_rows {
            write_u32(&mut bytes, *prefix1);
            write_u32(&mut bytes, *prefix2);
            write_u32(&mut bytes, *prefix3);
            write_u32(&mut bytes, *start);
            write_u32(&mut bytes, *len);
        }
        for (token_id, count, score) in &candidates {
            write_u32(&mut bytes, *token_id);
            write_u32(&mut bytes, *count);
            if !compact_count_records {
                write_i32(&mut bytes, *score);
            }
        }
        bytes.extend_from_slice(&token_bytes);
        bytes
    }

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
