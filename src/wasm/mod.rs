use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen::prelude::*;
use web_sys::Performance;

use crate::ObadhEngine;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn start() {
    // Always set the panic hook - it's crucial for debugging
    console_error_panic_hook::set_once();

    // Log initialization
    web_sys::console::log_1(&"Obadh Engine WASM module initializing...".into());
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
        web_sys::console::log_1(&"Creating new ObadhaWasm instance...".into());
        Self {
            engine: ObadhEngine::new(),
        }
    }

    /// Transliterate text from Roman to Bengali
    #[wasm_bindgen]
    pub fn transliterate(&self, text: &str) -> String {
        // For empty text, return empty string immediately
        if text.trim().is_empty() {
            return String::new();
        }

        self.engine.transliterate(text)
    }

    /// Transliterate with options for debug/verbose output
    #[wasm_bindgen]
    pub fn transliterate_with_options(
        &self,
        text: &str,
        options_js: JsValue,
    ) -> Result<JsValue, JsValue> {
        // For empty text, return a basic result immediately
        if text.trim().is_empty() {
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
