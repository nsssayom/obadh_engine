//! Stable feature extraction contracts for ML layers above Obadh.
//!
//! The deterministic transliterator remains the hot path. This module exposes
//! a versioned structural view that training and inference adapters can consume
//! without scraping debug JSON or final Bengali text.

mod features;

pub use features::{
    extract_features, MlFeatureDocument, MlFeatureSlot, MlPhoneticUnitFeatures, MlTokenFeatures,
    FEATURE_SCHEMA_VERSION, FEATURE_SLOTS_PER_UNIT,
};
