//! Content fingerprints for pinned runtime artifacts.
//!
//! Downstreams pin the FST and n-gram artifacts (`bn.fst`, the loanword FST, the
//! autosuggest LM) in their app resources, versioned separately from this crate.
//! A fingerprint lets them detect a stale or mismatched artifact **loudly at
//! load time** instead of silently degrading — the failure mode a keyboard
//! integrator actually hits when a pinned artifact drifts from the engine
//! version.
//!
//! The fingerprint is an FNV-1a hash over the whole artifact byte image. It is
//! deliberately not cryptographic: it defends against accidental staleness (the
//! wrong revision shipped), not an adversary crafting a collision. In exchange
//! it is cheap enough to compute at load and needs no hashing dependency, so it
//! is available on every target including `wasm32`.

use std::fmt;

/// FNV-1a fingerprint over an artifact's raw bytes.
///
/// Stable across platforms and crate versions: identical bytes always hash to
/// the same value, and any single-byte difference changes it. Never returns 0,
/// so a caller may reserve 0 to mean "unknown".
pub fn artifact_fingerprint(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET;
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(PRIME);
    }

    if hash == 0 {
        1
    } else {
        hash
    }
}

/// A loaded (or about-to-load) artifact's fingerprint did not match the value
/// the caller expected for its pinned crate version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FingerprintMismatch {
    pub expected: u64,
    pub actual: u64,
}

impl fmt::Display for FingerprintMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "artifact fingerprint mismatch: expected {:#018x}, found {:#018x} \
             (stale or wrong artifact for this engine version?)",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for FingerprintMismatch {}

/// Verify raw artifact bytes against an expected fingerprint before loading.
///
/// Returns loudly on mismatch so a stale pinned artifact fails fast at load
/// rather than degrading suggestions silently. The expected value comes from the
/// crate ↔ artifact compatibility table published in the release notes.
pub fn verify_artifact_fingerprint(bytes: &[u8], expected: u64) -> Result<(), FingerprintMismatch> {
    let actual = artifact_fingerprint(bytes);
    if actual == expected {
        Ok(())
    } else {
        Err(FingerprintMismatch { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_and_nonzero() {
        assert_eq!(
            artifact_fingerprint(b"obadh"),
            artifact_fingerprint(b"obadh")
        );
        assert_ne!(artifact_fingerprint(b""), 0);
        // FNV-1a of the empty input is the offset basis.
        assert_eq!(artifact_fingerprint(b""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn fingerprint_changes_on_any_byte_difference() {
        assert_ne!(
            artifact_fingerprint(b"bn.fst.v1"),
            artifact_fingerprint(b"bn.fst.v2")
        );
        assert_ne!(
            artifact_fingerprint(&[0, 1, 2]),
            artifact_fingerprint(&[0, 2, 1])
        );
    }

    #[test]
    fn verify_passes_on_match_and_reports_both_sides_on_mismatch() {
        let bytes = b"artifact bytes";
        let good = artifact_fingerprint(bytes);
        assert!(verify_artifact_fingerprint(bytes, good).is_ok());

        let error = verify_artifact_fingerprint(bytes, good ^ 1).unwrap_err();
        assert_eq!(error.expected, good ^ 1);
        assert_eq!(error.actual, good);
    }
}
