use std::fmt;

pub(crate) const LEXICON_MAGIC_V1: &[u8; 8] = b"OBACLEX1";
pub(crate) const LEXICON_MAGIC_V2: &[u8; 8] = b"OBACLEX2";
pub(crate) const LEXICON_MAGIC_V3: &[u8; 8] = b"OBACLEX3";
pub(crate) const LEXICON_MAGIC: &[u8; 8] = LEXICON_MAGIC_V3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LexiconArtifactVersion {
    V1,
    V2,
    V3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexiconArtifactError {
    InvalidMagic,
    Truncated,
    TrailingBytes,
    InvalidUtf8,
    EmptyWord { index: usize },
    DuplicateWord { index: usize },
    UnsortedWord { index: usize },
    WordTooLong { bytes: usize },
    SkeletonTooLong { bytes: usize },
    TooManySkeletons { skeletons: usize },
    InvalidSkeletonIndex { index: usize, skeleton_index: u32 },
    TooManyEntries { entries: usize },
}

impl fmt::Display for LexiconArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(formatter, "invalid autocorrect lexicon artifact header"),
            Self::Truncated => write!(formatter, "truncated autocorrect lexicon artifact"),
            Self::TrailingBytes => {
                write!(formatter, "trailing bytes in autocorrect lexicon artifact")
            }
            Self::InvalidUtf8 => write!(
                formatter,
                "invalid UTF-8 word in autocorrect lexicon artifact"
            ),
            Self::EmptyWord { index } => {
                write!(formatter, "empty word at autocorrect lexicon entry {index}")
            }
            Self::DuplicateWord { index } => {
                write!(
                    formatter,
                    "duplicate word at autocorrect lexicon entry {index}"
                )
            }
            Self::UnsortedWord { index } => {
                write!(
                    formatter,
                    "unsorted word at autocorrect lexicon entry {index}"
                )
            }
            Self::WordTooLong { bytes } => {
                write!(
                    formatter,
                    "autocorrect lexicon word is too long for artifact format: {bytes} bytes"
                )
            }
            Self::SkeletonTooLong { bytes } => {
                write!(
                    formatter,
                    "autocorrect lexicon skeleton is too long for artifact format: {bytes} bytes"
                )
            }
            Self::TooManySkeletons { skeletons } => {
                write!(
                    formatter,
                    "autocorrect lexicon has too many skeletons for artifact format: {skeletons}"
                )
            }
            Self::InvalidSkeletonIndex {
                index,
                skeleton_index,
            } => {
                write!(
                    formatter,
                    "invalid skeleton index {skeleton_index} at autocorrect lexicon entry {index}"
                )
            }
            Self::TooManyEntries { entries } => {
                write!(
                    formatter,
                    "autocorrect lexicon has too many entries for artifact format: {entries}"
                )
            }
        }
    }
}

impl std::error::Error for LexiconArtifactError {}

pub(crate) struct ArtifactReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ArtifactReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(crate) fn read_magic(&mut self) -> Result<LexiconArtifactVersion, LexiconArtifactError> {
        let magic = self.read_exact(LEXICON_MAGIC.len())?;
        match magic {
            _ if magic == LEXICON_MAGIC_V1 => Ok(LexiconArtifactVersion::V1),
            _ if magic == LEXICON_MAGIC_V2 => Ok(LexiconArtifactVersion::V2),
            _ if magic == LEXICON_MAGIC_V3 => Ok(LexiconArtifactVersion::V3),
            _ => Err(LexiconArtifactError::InvalidMagic),
        }
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16, LexiconArtifactError> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, LexiconArtifactError> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn read_word(&mut self, len: usize) -> Result<&'a str, LexiconArtifactError> {
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes).map_err(|_| LexiconArtifactError::InvalidUtf8)
    }

    pub(crate) fn finish(self) -> Result<(), LexiconArtifactError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(LexiconArtifactError::TrailingBytes)
        }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], LexiconArtifactError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(LexiconArtifactError::Truncated)?;
        if end > self.bytes.len() {
            return Err(LexiconArtifactError::Truncated);
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }
}

pub(crate) fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
