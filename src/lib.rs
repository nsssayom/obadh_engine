//! Obadh Engine - A linguistically accurate Roman to Bengali transliteration engine
//!
//! This library provides a transliteration engine for converting Roman script
//! to Bengali script, focusing on accuracy and linguistic correctness.

pub mod definitions;
pub mod engine;
pub mod ml;
pub mod wasm;

// Re-export commonly used types for convenience
pub use engine::{PhoneticUnit, PhoneticUnitType, Token, TokenType, Tokenizer};
pub use engine::{SanitizeResult, Sanitizer};
pub use ml::{
    MlFeatureDocument, MlFeatureSlot, MlPhoneticUnitFeatures, MlTokenFeatures,
    FEATURE_SCHEMA_VERSION, FEATURE_SLOTS_PER_UNIT,
};
pub use wasm::ObadhaWasm;

/// Main entry point for the Obadh transliteration engine
pub struct ObadhEngine {
    transliterator: engine::Transliterator,
}

impl ObadhEngine {
    /// Create a new engine with default settings
    pub fn new() -> Self {
        Self {
            transliterator: engine::Transliterator::new(),
        }
    }

    /// Transliterate Roman text to Bengali
    pub fn transliterate(&self, text: &str) -> String {
        self.transliterator.transliterate(text)
    }

    /// Transliterate Roman text after dropping unsupported characters.
    pub fn transliterate_lenient(&self, text: &str) -> String {
        self.transliterator.transliterate_lenient(text)
    }

    /// Sanitize input text to ensure it contains only valid characters
    pub fn sanitize(&self, text: &str) -> SanitizeResult {
        self.transliterator.sanitize(text)
    }

    /// Tokenize input text into words and other tokens
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        self.transliterator.tokenize(text)
    }

    /// Transliterate already-tokenized input to Bengali
    pub fn transliterate_tokens(&self, tokens: &[Token]) -> String {
        self.transliterator.transliterate_tokens(tokens)
    }

    /// Transliterate one token using its surrounding tokenized context.
    pub fn transliterate_token_at(&self, tokens: &[Token], index: usize) -> Option<String> {
        self.transliterator.transliterate_token_at(tokens, index)
    }

    /// Tokenize a word into phonetic units for Bengali transliteration
    pub fn tokenize_phonetic(&self, word: &str) -> Vec<PhoneticUnit> {
        self.transliterator.tokenize_phonetic(word)
    }

    /// Tokenize a word into a caller-owned phonetic-unit buffer.
    ///
    /// The buffer is cleared before use and then reused. Prefer this method for
    /// high-frequency editor or typing integrations that repeatedly analyze the
    /// active word.
    pub fn tokenize_phonetic_into(&self, word: &str, units: &mut Vec<PhoneticUnit>) {
        self.transliterator.tokenize_phonetic_into(word, units);
    }

    /// Get a new tokenizer instance for custom tokenization
    pub fn get_tokenizer(&self) -> Tokenizer {
        Tokenizer::new()
    }

    /// Extract versioned structural features for ML layers above Obadh.
    ///
    /// This API is intentionally separate from the deterministic transliteration
    /// hot path. Unsupported input is reported instead of cleaned so training
    /// pipelines can skip or audit rejected rows explicitly.
    pub fn ml_features(&self, text: &str) -> MlFeatureDocument {
        ml::extract_features(&self.transliterator, text)
    }
}

impl Default for ObadhEngine {
    fn default() -> Self {
        Self::new()
    }
}
