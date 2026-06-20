use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen::prelude::*;
use web_sys::Performance;

use crate::{
    AutocorrectConfig, AutocorrectDecision, AutocorrectEngine, CandidateFeatures,
    CharCandidateReranker, CorrectionCandidate, CorrectionSource, Lexicon, LexiconEntry,
    LexiconStats, ObadhEngine, ScoredCorrectionCandidate,
};

const AUTOCORRECT_RERANK_POOL_SIZE: usize = 512;
const AUTOCORRECT_RESPONSE_CANDIDATES: usize = 24;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// Helper function to get the performance object
fn get_performance() -> Option<Performance> {
    web_sys::window()?.performance()
}

// Helper function to measure time
fn now() -> f64 {
    get_performance().map_or(0.0, |performance| performance.now())
}

/// Output options for the transliteration
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct TransliterationOptions {
    /// Output performance metrics
    pub debug: bool,
    /// Include token analysis in output
    pub verbose: bool,
}

#[wasm_bindgen]
impl TransliterationOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            debug: false,
            verbose: false,
        }
    }
}

impl Default for TransliterationOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for the transliteration process
#[derive(Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_ms: f64,
    pub sanitize_ms: f64,
    pub tokenize_ms: f64,
    pub transliterate_ms: f64,
}

/// Detailed token analysis for verbose output
#[derive(Serialize, Deserialize)]
pub struct TokenAnalysis {
    pub content: String,
    pub position: usize,
    pub r#type: String, // "type" is a reserved keyword in JS
    pub transliterated: Option<String>,
    pub phonetic_units: Option<Vec<PhoneticUnitInfo>>,
}

/// Information about a phonetic unit
#[derive(Serialize, Deserialize)]
pub struct PhoneticUnitInfo {
    pub text: String,
    pub position: usize,
    pub r#type: String, // "type" is a reserved keyword in JS
}

/// Complete transliteration result
#[derive(Serialize, Deserialize)]
pub struct TransliterationResult {
    pub input: String,
    pub output: String,
    pub performance: Option<PerformanceMetrics>,
    pub token_analysis: Option<Vec<TokenAnalysis>>,
}

#[derive(Serialize)]
pub struct AutocorrectLexiconStats {
    pub entries: usize,
    pub trie_nodes: usize,
    pub trie_edges: usize,
    pub skeleton_keys: usize,
    pub skeleton_delete_keys: usize,
}

#[derive(Serialize)]
pub struct AutocorrectCandidateInfo {
    pub text: String,
    pub source: &'static str,
    pub edit_cost: u16,
    pub frequency: u32,
    pub score: i32,
    pub model_score: Option<f32>,
    pub features: [i16; crate::AUTOCORRECT_FEATURE_DIM],
}

#[derive(Serialize)]
pub struct AutocorrectLabResult {
    pub roman_input: String,
    pub obadh_output: String,
    pub input: String,
    pub elapsed_ms: f64,
    pub replacement: Option<AutocorrectCandidateInfo>,
    pub candidates: Vec<AutocorrectCandidateInfo>,
    pub lexicon: AutocorrectLexiconStats,
}

/// ObdahWasm is the main WASM interface to the Obadh engine
#[wasm_bindgen]
pub struct ObadhaWasm {
    engine: ObadhEngine,
}

#[wasm_bindgen]
impl ObadhaWasm {
    /// Create a new instance of the Obadh engine
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            engine: ObadhEngine::new(),
        }
    }

    /// Transliterate text from Roman to Bengali
    #[wasm_bindgen]
    pub fn transliterate(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        self.engine.transliterate(text)
    }

    /// Transliterate text after dropping unsupported characters
    #[wasm_bindgen]
    pub fn transliterate_lenient(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        self.engine.transliterate_lenient(text)
    }

    /// Transliterate with options for debug/verbose output
    #[wasm_bindgen]
    pub fn transliterate_with_options(
        &self,
        text: &str,
        options_js: JsValue,
    ) -> Result<JsValue, JsValue> {
        if text.is_empty() {
            let empty_result = TransliterationResult {
                input: String::new(),
                output: String::new(),
                performance: None,
                token_analysis: None,
            };
            return Ok(to_value(&empty_result)?);
        }

        // Convert JS options to Rust struct
        let options: TransliterationOptions = match from_value(options_js) {
            Ok(opts) => opts,
            Err(e) => {
                return Err(JsValue::from_str(&format!(
                    "Failed to parse options: {}",
                    e
                )));
            }
        };

        // Create result object
        let mut result = TransliterationResult {
            input: text.to_string(),
            output: String::new(),
            performance: None,
            token_analysis: None,
        };

        // For debug/verbose modes, measure performance
        if options.debug || options.verbose {
            // Measure sanitization performance
            let sanitize_start = now();
            let sanitized = self.engine.sanitize(text);
            let sanitize_duration = now() - sanitize_start;
            let Ok(sanitized) = sanitized else {
                result.output = text.to_string();
                result.performance = Some(PerformanceMetrics {
                    sanitize_ms: sanitize_duration,
                    tokenize_ms: 0.0,
                    transliterate_ms: 0.0,
                    total_ms: sanitize_duration,
                });
                if options.verbose {
                    result.token_analysis = Some(Vec::new());
                }
                return Ok(to_value(&result)?);
            };

            // Measure tokenization performance
            let tokenize_start = now();
            let tokens = self.engine.tokenize(&sanitized);
            let tokenize_duration = now() - tokenize_start;

            // Measure transliteration performance
            let transliterate_start = now();
            result.output = self.engine.transliterate_tokens(&tokens);
            let transliterate_duration = now() - transliterate_start;

            // Calculate total duration
            let total_duration = sanitize_duration + tokenize_duration + transliterate_duration;

            // Populate performance metrics with actual measurements
            result.performance = Some(PerformanceMetrics {
                sanitize_ms: sanitize_duration,
                tokenize_ms: tokenize_duration,
                transliterate_ms: transliterate_duration,
                total_ms: total_duration,
            });

            // Add token analysis if verbose is enabled
            if options.verbose {
                let mut token_analysis = Vec::new();

                for (index, token) in tokens.iter().enumerate() {
                    let mut analysis = TokenAnalysis {
                        content: token.content.clone(),
                        position: token.position,
                        r#type: format!("{:?}", token.token_type),
                        transliterated: self.engine.transliterate_token_at(&tokens, index),
                        phonetic_units: None,
                    };

                    // Add phonetic units for Word tokens
                    if let crate::TokenType::Word = token.token_type {
                        let phonetic_units = self.engine.tokenize_phonetic(&token.content);

                        if !phonetic_units.is_empty() {
                            let mut units_info = Vec::new();

                            for unit in phonetic_units {
                                units_info.push(PhoneticUnitInfo {
                                    text: unit.text.clone(),
                                    position: unit.position,
                                    r#type: format!("{:?}", unit.unit_type),
                                });
                            }

                            analysis.phonetic_units = Some(units_info);
                        }
                    }

                    token_analysis.push(analysis);
                }

                result.token_analysis = Some(token_analysis);
            }
        } else {
            // Simple transliteration without metrics
            result.output = self.engine.transliterate(text);
        }

        // Convert to JsValue and return
        match to_value(&result) {
            Ok(val) => Ok(val),
            Err(e) => Err(JsValue::from_str(&format!(
                "Failed to serialize result: {}",
                e
            ))),
        }
    }

    /// Get version information
    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

impl Default for ObadhaWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub struct ObadhAutocorrectWasm {
    obadh: ObadhEngine,
    engine: AutocorrectEngine,
    char_reranker: Option<CharCandidateReranker>,
    stats: LexiconStats,
}

#[wasm_bindgen]
impl ObadhAutocorrectWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_lexicon(Lexicon::default(), None)
    }

    #[wasm_bindgen(js_name = fromCompactLexicon)]
    pub fn from_compact_lexicon(bytes: &[u8]) -> Result<ObadhAutocorrectWasm, JsValue> {
        let lexicon = Lexicon::from_compact_bytes(bytes)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self::from_lexicon(lexicon, None))
    }

    #[wasm_bindgen(js_name = fromCompactLexiconAndCharReranker)]
    pub fn from_compact_lexicon_and_char_reranker(
        lexicon_bytes: &[u8],
        reranker_json: &[u8],
    ) -> Result<ObadhAutocorrectWasm, JsValue> {
        let lexicon = Lexicon::from_compact_bytes(lexicon_bytes)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let char_reranker = CharCandidateReranker::from_json_bytes(reranker_json)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self::from_lexicon(lexicon, Some(char_reranker)))
    }

    #[wasm_bindgen(js_name = fromTsv)]
    pub fn from_tsv(lexicon_tsv: &str) -> Result<ObadhAutocorrectWasm, JsValue> {
        let entries = parse_lexicon_tsv(lexicon_tsv).map_err(|error| JsValue::from_str(&error))?;
        Ok(Self::from_lexicon(Lexicon::new(entries), None))
    }

    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        to_value(&AutocorrectLexiconStats::from(self.stats))
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen]
    pub fn suggest(&self, roman_input: &str) -> Result<JsValue, JsValue> {
        let start = now();
        if roman_input.trim().is_empty() {
            let empty_result = AutocorrectLabResult {
                roman_input: String::new(),
                obadh_output: String::new(),
                input: String::new(),
                elapsed_ms: now() - start,
                replacement: None,
                candidates: Vec::new(),
                lexicon: AutocorrectLexiconStats::from(self.stats),
            };
            return to_value(&empty_result).map_err(|error| JsValue::from_str(&error.to_string()));
        }

        let request = self.obadh.autocorrect_request(roman_input);
        let mut decision = self.engine.decide(request);
        let ranked_candidates = if let Some(char_reranker) = &self.char_reranker {
            let ranked = char_reranker.rank_candidates(roman_input, &decision.candidates);
            decision.replacement =
                char_reranker.calibrated_replacement_from_ranked_candidates(&ranked);
            decision.candidates = ranked
                .iter()
                .map(|scored| scored.candidate.clone())
                .collect();
            Some(ranked)
        } else {
            None
        };
        let result = autocorrect_result(
            roman_input,
            decision,
            now() - start,
            self.stats,
            ranked_candidates.as_deref(),
        );
        to_value(&result).map_err(|error| JsValue::from_str(&error.to_string()))
    }
}

impl Default for ObadhAutocorrectWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl ObadhAutocorrectWasm {
    fn from_lexicon(lexicon: Lexicon, char_reranker: Option<CharCandidateReranker>) -> Self {
        let stats = lexicon.stats();
        Self {
            obadh: ObadhEngine::new(),
            engine: AutocorrectEngine::with_config(lexicon, autocorrect_lab_config()),
            char_reranker,
            stats,
        }
    }
}

impl From<LexiconStats> for AutocorrectLexiconStats {
    fn from(stats: LexiconStats) -> Self {
        Self {
            entries: stats.entries,
            trie_nodes: stats.trie_nodes,
            trie_edges: stats.trie_edges,
            skeleton_keys: stats.skeleton_keys,
            skeleton_delete_keys: stats.skeleton_delete_keys,
        }
    }
}

fn autocorrect_lab_config() -> AutocorrectConfig {
    AutocorrectConfig {
        max_candidates: AUTOCORRECT_RERANK_POOL_SIZE,
        search_known_input: true,
        max_skeleton_candidates: AUTOCORRECT_RERANK_POOL_SIZE,
        ..AutocorrectConfig::default()
    }
}

fn autocorrect_result(
    roman_input: &str,
    decision: AutocorrectDecision,
    elapsed_ms: f64,
    stats: LexiconStats,
    ranked_candidates: Option<&[ScoredCorrectionCandidate]>,
) -> AutocorrectLabResult {
    AutocorrectLabResult {
        roman_input: roman_input.to_string(),
        obadh_output: decision.input.clone(),
        input: decision.input,
        elapsed_ms,
        replacement: decision.replacement.as_ref().map(|candidate| {
            autocorrect_candidate_info(candidate, cached_model_score(ranked_candidates, candidate))
        }),
        candidates: autocorrect_candidate_infos(&decision.candidates, ranked_candidates),
        lexicon: AutocorrectLexiconStats::from(stats),
    }
}

fn autocorrect_candidate_infos(
    candidates: &[CorrectionCandidate],
    ranked_candidates: Option<&[ScoredCorrectionCandidate]>,
) -> Vec<AutocorrectCandidateInfo> {
    if let Some(ranked_candidates) = ranked_candidates {
        return ranked_candidates
            .iter()
            .take(AUTOCORRECT_RESPONSE_CANDIDATES)
            .map(|scored| autocorrect_scored_candidate_info(scored))
            .collect();
    }

    candidates
        .iter()
        .take(AUTOCORRECT_RESPONSE_CANDIDATES)
        .map(|candidate| autocorrect_candidate_info(candidate, None))
        .collect()
}

fn autocorrect_scored_candidate_info(
    scored: &ScoredCorrectionCandidate,
) -> AutocorrectCandidateInfo {
    autocorrect_candidate_info(&scored.candidate, Some(scored.model_score))
}

fn autocorrect_candidate_info(
    candidate: &CorrectionCandidate,
    model_score: Option<f32>,
) -> AutocorrectCandidateInfo {
    AutocorrectCandidateInfo {
        text: candidate.text.clone(),
        source: correction_source_name(candidate.source),
        edit_cost: candidate.edit_cost.0,
        frequency: candidate.frequency,
        score: candidate.score,
        model_score,
        features: candidate_features_array(candidate.features),
    }
}

fn cached_model_score(
    ranked_candidates: Option<&[ScoredCorrectionCandidate]>,
    candidate: &CorrectionCandidate,
) -> Option<f32> {
    ranked_candidates?.iter().find_map(|scored| {
        (scored.candidate.text == candidate.text && scored.candidate.source == candidate.source)
            .then_some(scored.model_score)
    })
}

fn correction_source_name(source: CorrectionSource) -> &'static str {
    match source {
        CorrectionSource::NoChange => "no_change",
        CorrectionSource::LexiconEdit => "lexicon_edit",
        CorrectionSource::PhoneticSkeleton => "phonetic_skeleton",
        CorrectionSource::RomanEdit => "roman_edit",
    }
}

fn candidate_features_array(features: CandidateFeatures) -> [i16; crate::AUTOCORRECT_FEATURE_DIM] {
    features.as_i16_array()
}

fn parse_lexicon_tsv(lexicon_tsv: &str) -> Result<Vec<LexiconEntry>, String> {
    let mut entries = Vec::new();

    for (line_index, line) in lexicon_tsv.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split('\t');
        let word = parts.next().unwrap_or_default().trim();
        if word.is_empty() {
            return Err(format!("empty lexicon word at line {line_number}"));
        }

        let frequency = match parts.next().map(str::trim).filter(|part| !part.is_empty()) {
            Some(raw) => raw
                .parse::<u32>()
                .map_err(|error| format!("invalid frequency at line {line_number}: {error}"))?,
            None => 1,
        };

        if parts.next().is_some() {
            return Err(format!("expected word<TAB>frequency at line {line_number}"));
        }

        entries.push(LexiconEntry::new(word, frequency));
    }

    if entries.is_empty() {
        return Err("lexicon TSV did not contain any entries".to_string());
    }

    Ok(entries)
}
