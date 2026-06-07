//! Definitions for Bengali conjuncts
//!
//! This module provides a comprehensive set of Bengali conjunct definitions
//! based on phonetic components. The engine uses this compiled Rust rule table
//! directly; source CSV data is not parsed or shipped on the runtime path.

use crate::definitions::consonant_value;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Structure to store and manage Bengali conjunct definitions.
#[derive(Debug)]
pub struct ConjunctDefinitions {
    /// Trie for allocation-free conjunct lookup from component parts.
    conjunct_trie: ConjunctTrie,
}

#[derive(Debug, Clone, Copy)]
struct ConjunctRule {
    key: &'static str,
    value: &'static str,
}

const CONJUNCT_RULES: &[ConjunctRule] = &[
    ConjunctRule {
        key: "kk",
        value: "ক্ক",
    },
    ConjunctRule {
        key: "kT",
        value: "ক্ট",
    },
    ConjunctRule {
        key: "kTr",
        value: "ক্ট্র",
    },
    ConjunctRule {
        key: "kt",
        value: "ক্ত",
    },
    ConjunctRule {
        key: "ktr",
        value: "ক্ত্র",
    },
    ConjunctRule {
        key: "kn",
        value: "ক্ন",
    },
    ConjunctRule {
        key: "kw",
        value: "ক্ব",
    },
    ConjunctRule {
        key: "km",
        value: "ক্ম",
    },
    ConjunctRule {
        key: "ky",
        value: "ক্য",
    },
    ConjunctRule {
        key: "kY",
        value: "ক্য",
    },
    ConjunctRule {
        key: "kr",
        value: "ক্র",
    },
    ConjunctRule {
        key: "kl",
        value: "ক্ল",
    },
    ConjunctRule {
        key: "kSh",
        value: "ক্ষ",
    },
    ConjunctRule {
        key: "ksh",
        value: "ক্ষ",
    },
    ConjunctRule {
        key: "kkh",
        value: "ক্ষ",
    },
    ConjunctRule {
        key: "kShN",
        value: "ক্ষ্ণ",
    },
    ConjunctRule {
        key: "kshN",
        value: "ক্ষ্ণ",
    },
    ConjunctRule {
        key: "kkhN",
        value: "ক্ষ্ণ",
    },
    ConjunctRule {
        key: "kShw",
        value: "ক্ষ্ব",
    },
    ConjunctRule {
        key: "kshw",
        value: "ক্ষ্ব",
    },
    ConjunctRule {
        key: "kkhw",
        value: "ক্ষ্ব",
    },
    ConjunctRule {
        key: "kShm",
        value: "ক্ষ্ম",
    },
    ConjunctRule {
        key: "kshm",
        value: "ক্ষ্ম",
    },
    ConjunctRule {
        key: "kkhm",
        value: "ক্ষ্ম",
    },
    ConjunctRule {
        key: "kShy",
        value: "ক্ষ্য",
    },
    ConjunctRule {
        key: "kShY",
        value: "ক্ষ্য",
    },
    ConjunctRule {
        key: "kshy",
        value: "ক্ষ্য",
    },
    ConjunctRule {
        key: "kkhy",
        value: "ক্ষ্য",
    },
    ConjunctRule {
        key: "kkhY",
        value: "ক্ষ্য",
    },
    ConjunctRule {
        key: "kShmy",
        value: "ক্ষ্ম্য",
    },
    ConjunctRule {
        key: "kshmy",
        value: "ক্ষ্ম্য",
    },
    ConjunctRule {
        key: "kkhmy",
        value: "ক্ষ্ম্য",
    },
    ConjunctRule {
        key: "ks",
        value: "ক্স",
    },
    ConjunctRule {
        key: "khy",
        value: "খ্য",
    },
    ConjunctRule {
        key: "khY",
        value: "খ্য",
    },
    ConjunctRule {
        key: "khr",
        value: "খ্র",
    },
    ConjunctRule {
        key: "gN",
        value: "গ্‌ণ",
    },
    ConjunctRule {
        key: "gdh",
        value: "গ্ধ",
    },
    ConjunctRule {
        key: "gdhy",
        value: "গ্ধ্য",
    },
    ConjunctRule {
        key: "gdhr",
        value: "গ্ধ্র",
    },
    ConjunctRule {
        key: "gn",
        value: "গ্ন",
    },
    ConjunctRule {
        key: "gny",
        value: "গ্ন্য",
    },
    ConjunctRule {
        key: "gw",
        value: "গ্ব",
    },
    ConjunctRule {
        key: "gm",
        value: "গ্ম",
    },
    ConjunctRule {
        key: "gy",
        value: "গ্য",
    },
    ConjunctRule {
        key: "gY",
        value: "গ্য",
    },
    ConjunctRule {
        key: "gr",
        value: "গ্র",
    },
    ConjunctRule {
        key: "gry",
        value: "গ্র্য",
    },
    ConjunctRule {
        key: "gl",
        value: "গ্ল",
    },
    ConjunctRule {
        key: "ghn",
        value: "ঘ্ন",
    },
    ConjunctRule {
        key: "ghy",
        value: "ঘ্য",
    },
    ConjunctRule {
        key: "ghY",
        value: "ঘ্য",
    },
    ConjunctRule {
        key: "ghr",
        value: "ঘ্র",
    },
    ConjunctRule {
        key: "Ngk",
        value: "ঙ্ক",
    },
    ConjunctRule {
        key: "Ngkt",
        value: "ঙ্‌ক্ত",
    },
    ConjunctRule {
        key: "Ngky",
        value: "ঙ্ক্য",
    },
    ConjunctRule {
        key: "NgkY",
        value: "ঙ্ক্য",
    },
    ConjunctRule {
        key: "NgkSh",
        value: "ঙ্ক্ষ",
    },
    ConjunctRule {
        key: "Ngksh",
        value: "ঙ্ক্ষ",
    },
    ConjunctRule {
        key: "Ngkh",
        value: "ঙ্খ",
    },
    ConjunctRule {
        key: "Ngkhy",
        value: "ঙ্খ্য",
    },
    ConjunctRule {
        key: "NgkhY",
        value: "ঙ্খ্য",
    },
    ConjunctRule {
        key: "Ngg",
        value: "ঙ্গ",
    },
    ConjunctRule {
        key: "Nggy",
        value: "ঙ্গ্য",
    },
    ConjunctRule {
        key: "NggY",
        value: "ঙ্গ্য",
    },
    ConjunctRule {
        key: "Nggh",
        value: "ঙ্ঘ",
    },
    ConjunctRule {
        key: "Ngghy",
        value: "ঙ্ঘ্য",
    },
    ConjunctRule {
        key: "NgghY",
        value: "ঙ্ঘ্য",
    },
    ConjunctRule {
        key: "Ngghr",
        value: "ঙ্ঘ্র",
    },
    ConjunctRule {
        key: "Ngm",
        value: "ঙ্ম",
    },
    ConjunctRule {
        key: "cc",
        value: "চ্চ",
    },
    ConjunctRule {
        key: "cch",
        value: "চ্ছ",
    },
    ConjunctRule {
        key: "cC",
        value: "চ্ছ",
    },
    ConjunctRule {
        key: "cCh",
        value: "চ্ছ",
    },
    ConjunctRule {
        key: "cChh",
        value: "চ্ছ",
    },
    ConjunctRule {
        key: "cchw",
        value: "চ্ছ্ব",
    },
    ConjunctRule {
        key: "cCw",
        value: "চ্ছ্ব",
    },
    ConjunctRule {
        key: "cChw",
        value: "চ্ছ্ব",
    },
    ConjunctRule {
        key: "cChhw",
        value: "চ্ছ্ব",
    },
    ConjunctRule {
        key: "cchr",
        value: "চ্ছ্র",
    },
    ConjunctRule {
        key: "cCr",
        value: "চ্ছ্র",
    },
    ConjunctRule {
        key: "cChr",
        value: "চ্ছ্র",
    },
    ConjunctRule {
        key: "cChhr",
        value: "চ্ছ্র",
    },
    ConjunctRule {
        key: "cNG",
        value: "চ্ঞ",
    },
    ConjunctRule {
        key: "cw",
        value: "চ্ব",
    },
    ConjunctRule {
        key: "cy",
        value: "চ্য",
    },
    ConjunctRule {
        key: "cY",
        value: "চ্য",
    },
    ConjunctRule {
        key: "jj",
        value: "জ্জ",
    },
    ConjunctRule {
        key: "Jj",
        value: "জ্জ",
    },
    ConjunctRule {
        key: "jJ",
        value: "জ্জ",
    },
    ConjunctRule {
        key: "JJ",
        value: "জ্জ",
    },
    ConjunctRule {
        key: "jjw",
        value: "জ্জ্ব",
    },
    ConjunctRule {
        key: "Jjw",
        value: "জ্জ্ব",
    },
    ConjunctRule {
        key: "jJw",
        value: "জ্জ্ব",
    },
    ConjunctRule {
        key: "JJw",
        value: "জ্জ্ব",
    },
    ConjunctRule {
        key: "jjh",
        value: "জ্ঝ",
    },
    ConjunctRule {
        key: "Jjh",
        value: "জ্ঝ",
    },
    ConjunctRule {
        key: "jNG",
        value: "জ্ঞ",
    },
    ConjunctRule {
        key: "JNG",
        value: "জ্ঞ",
    },
    ConjunctRule {
        key: "jn",
        value: "জ্ঞ",
    },
    ConjunctRule {
        key: "Jn",
        value: "জ্ঞ",
    },
    ConjunctRule {
        key: "jw",
        value: "জ্ব",
    },
    ConjunctRule {
        key: "Jw",
        value: "জ্ব",
    },
    ConjunctRule {
        key: "jy",
        value: "জ্য",
    },
    ConjunctRule {
        key: "Jy",
        value: "জ্য",
    },
    ConjunctRule {
        key: "jY",
        value: "জ্য",
    },
    ConjunctRule {
        key: "JY",
        value: "জ্য",
    },
    ConjunctRule {
        key: "jr",
        value: "জ্র",
    },
    ConjunctRule {
        key: "Jr",
        value: "জ্র",
    },
    ConjunctRule {
        key: "NGc",
        value: "ঞ্চ",
    },
    ConjunctRule {
        key: "NGch",
        value: "ঞ্ছ",
    },
    ConjunctRule {
        key: "NGj",
        value: "ঞ্জ",
    },
    ConjunctRule {
        key: "NGJ",
        value: "ঞ্জ",
    },
    ConjunctRule {
        key: "NGjh",
        value: "ঞ্ঝ",
    },
    ConjunctRule {
        key: "TT",
        value: "ট্ট",
    },
    ConjunctRule {
        key: "Tw",
        value: "ট্ব",
    },
    ConjunctRule {
        key: "Tm",
        value: "ট্ম",
    },
    ConjunctRule {
        key: "Ty",
        value: "ট্য",
    },
    ConjunctRule {
        key: "TY",
        value: "ট্য",
    },
    ConjunctRule {
        key: "Tr",
        value: "ট্র",
    },
    ConjunctRule {
        key: "DD",
        value: "ড্ড",
    },
    ConjunctRule {
        key: "Dw",
        value: "ড্ব",
    },
    ConjunctRule {
        key: "Dm",
        value: "ড্ম",
    },
    ConjunctRule {
        key: "Dy",
        value: "ড্য",
    },
    ConjunctRule {
        key: "DY",
        value: "ড্য",
    },
    ConjunctRule {
        key: "Dr",
        value: "ড্র",
    },
    ConjunctRule {
        key: "Rg",
        value: "ড়্গ",
    },
    ConjunctRule {
        key: "Dhy",
        value: "ঢ্য",
    },
    ConjunctRule {
        key: "DhY",
        value: "ঢ্য",
    },
    ConjunctRule {
        key: "Dhr",
        value: "ঢ্র",
    },
    ConjunctRule {
        key: "NT",
        value: "ণ্ট",
    },
    ConjunctRule {
        key: "NTh",
        value: "ণ্ঠ",
    },
    ConjunctRule {
        key: "NThy",
        value: "ণ্ঠ্য",
    },
    ConjunctRule {
        key: "NThY",
        value: "ণ্ঠ্য",
    },
    ConjunctRule {
        key: "ND",
        value: "ণ্ড",
    },
    ConjunctRule {
        key: "NDy",
        value: "ণ্ড্য",
    },
    ConjunctRule {
        key: "NDY",
        value: "ণ্ড্য",
    },
    ConjunctRule {
        key: "NDr",
        value: "ণ্ড্র",
    },
    ConjunctRule {
        key: "NDh",
        value: "ণ্ঢ",
    },
    ConjunctRule {
        key: "NN",
        value: "ণ্ণ",
    },
    ConjunctRule {
        key: "Nw",
        value: "ণ্ব",
    },
    ConjunctRule {
        key: "Nm",
        value: "ণ্ম",
    },
    ConjunctRule {
        key: "Ny",
        value: "ণ্য",
    },
    ConjunctRule {
        key: "NY",
        value: "ণ্য",
    },
    ConjunctRule {
        key: "tk",
        value: "ৎক",
    },
    ConjunctRule {
        key: "tkh",
        value: "ৎখ",
    },
    ConjunctRule {
        key: "tt",
        value: "ত্ত",
    },
    ConjunctRule {
        key: "ttr",
        value: "ত্ত্র",
    },
    ConjunctRule {
        key: "ttw",
        value: "ত্ত্ব",
    },
    ConjunctRule {
        key: "tty",
        value: "ত্ত্য",
    },
    ConjunctRule {
        key: "ttY",
        value: "ত্ত্য",
    },
    ConjunctRule {
        key: "tth",
        value: "ত্থ",
    },
    ConjunctRule {
        key: "tn",
        value: "ত্ন",
    },
    ConjunctRule {
        key: "tp",
        value: "ৎপ",
    },
    ConjunctRule {
        key: "tw",
        value: "ত্ব",
    },
    ConjunctRule {
        key: "tm",
        value: "ত্ম",
    },
    ConjunctRule {
        key: "tmy",
        value: "ত্ম্য",
    },
    ConjunctRule {
        key: "ty",
        value: "ত্য",
    },
    ConjunctRule {
        key: "tY",
        value: "ত্য",
    },
    ConjunctRule {
        key: "tr",
        value: "ত্র",
    },
    ConjunctRule {
        key: "try",
        value: "ত্র্য",
    },
    ConjunctRule {
        key: "tl",
        value: "ৎল",
    },
    ConjunctRule {
        key: "ts",
        value: "ৎস",
    },
    ConjunctRule {
        key: "thw",
        value: "থ্ব",
    },
    ConjunctRule {
        key: "thy",
        value: "থ্য",
    },
    ConjunctRule {
        key: "thY",
        value: "থ্য",
    },
    ConjunctRule {
        key: "thr",
        value: "থ্র",
    },
    ConjunctRule {
        key: "dg",
        value: "দ্গ",
    },
    ConjunctRule {
        key: "dgh",
        value: "দ্ঘ",
    },
    ConjunctRule {
        key: "dd",
        value: "দ্দ",
    },
    ConjunctRule {
        key: "ddw",
        value: "দ্দ্ব",
    },
    ConjunctRule {
        key: "ddh",
        value: "দ্ধ",
    },
    ConjunctRule {
        key: "dw",
        value: "দ্ব",
    },
    ConjunctRule {
        key: "dbh",
        value: "দ্ভ",
    },
    ConjunctRule {
        key: "dv",
        value: "দ্ভ",
    },
    ConjunctRule {
        key: "dbhr",
        value: "দ্ভ্র",
    },
    ConjunctRule {
        key: "dvr",
        value: "দ্ভ্র",
    },
    ConjunctRule {
        key: "dm",
        value: "দ্ম",
    },
    ConjunctRule {
        key: "dy",
        value: "দ্য",
    },
    ConjunctRule {
        key: "dY",
        value: "দ্য",
    },
    ConjunctRule {
        key: "dr",
        value: "দ্র",
    },
    ConjunctRule {
        key: "dry",
        value: "দ্র্য",
    },
    ConjunctRule {
        key: "dhn",
        value: "ধ্ন",
    },
    ConjunctRule {
        key: "dhw",
        value: "ধ্ব",
    },
    ConjunctRule {
        key: "dhm",
        value: "ধ্ম",
    },
    ConjunctRule {
        key: "dhy",
        value: "ধ্য",
    },
    ConjunctRule {
        key: "dhY",
        value: "ধ্য",
    },
    ConjunctRule {
        key: "dhr",
        value: "ধ্র",
    },
    ConjunctRule {
        key: "nT",
        value: "ন্ট",
    },
    ConjunctRule {
        key: "nTr",
        value: "ন্ট্র",
    },
    ConjunctRule {
        key: "nTh",
        value: "ন্ঠ",
    },
    ConjunctRule {
        key: "nD",
        value: "ন্ড",
    },
    ConjunctRule {
        key: "nDr",
        value: "ন্ড্র",
    },
    ConjunctRule {
        key: "nt",
        value: "ন্ত",
    },
    ConjunctRule {
        key: "ntw",
        value: "ন্ত্ব",
    },
    ConjunctRule {
        key: "nty",
        value: "ন্ত্য",
    },
    ConjunctRule {
        key: "ntY",
        value: "ন্ত্য",
    },
    ConjunctRule {
        key: "ntr",
        value: "ন্ত্র",
    },
    ConjunctRule {
        key: "ntry",
        value: "ন্ত্র্য",
    },
    ConjunctRule {
        key: "nth",
        value: "ন্থ",
    },
    ConjunctRule {
        key: "nthr",
        value: "ন্থ্র",
    },
    ConjunctRule {
        key: "nd",
        value: "ন্দ",
    },
    ConjunctRule {
        key: "ndy",
        value: "ন্দ্য",
    },
    ConjunctRule {
        key: "ndY",
        value: "ন্দ্য",
    },
    ConjunctRule {
        key: "ndw",
        value: "ন্দ্ব",
    },
    ConjunctRule {
        key: "ndr",
        value: "ন্দ্র",
    },
    ConjunctRule {
        key: "ndh",
        value: "ন্ধ",
    },
    ConjunctRule {
        key: "ndhy",
        value: "ন্ধ্য",
    },
    ConjunctRule {
        key: "ndhY",
        value: "ন্ধ্য",
    },
    ConjunctRule {
        key: "ndhr",
        value: "ন্ধ্র",
    },
    ConjunctRule {
        key: "nn",
        value: "ন্ন",
    },
    ConjunctRule {
        key: "nw",
        value: "ন্ব",
    },
    ConjunctRule {
        key: "nm",
        value: "ন্ম",
    },
    ConjunctRule {
        key: "ny",
        value: "ন্য",
    },
    ConjunctRule {
        key: "nY",
        value: "ন্য",
    },
    ConjunctRule {
        key: "pT",
        value: "প্ট",
    },
    ConjunctRule {
        key: "pt",
        value: "প্ত",
    },
    ConjunctRule {
        key: "pn",
        value: "প্ন",
    },
    ConjunctRule {
        key: "pp",
        value: "প্প",
    },
    ConjunctRule {
        key: "py",
        value: "প্য",
    },
    ConjunctRule {
        key: "pY",
        value: "প্য",
    },
    ConjunctRule {
        key: "pr",
        value: "প্র",
    },
    ConjunctRule {
        key: "pry",
        value: "প্র্য",
    },
    ConjunctRule {
        key: "pl",
        value: "প্ল",
    },
    ConjunctRule {
        key: "ps",
        value: "প্স",
    },
    ConjunctRule {
        key: "fr",
        value: "ফ্র",
    },
    ConjunctRule {
        key: "phr",
        value: "ফ্র",
    },
    ConjunctRule {
        key: "fl",
        value: "ফ্ল",
    },
    ConjunctRule {
        key: "phl",
        value: "ফ্ল",
    },
    ConjunctRule {
        key: "bj",
        value: "ব্জ",
    },
    ConjunctRule {
        key: "bJ",
        value: "ব্জ",
    },
    ConjunctRule {
        key: "bd",
        value: "ব্দ",
    },
    ConjunctRule {
        key: "bdh",
        value: "ব্ধ",
    },
    ConjunctRule {
        key: "bb",
        value: "ব্ব",
    },
    ConjunctRule {
        key: "bw",
        value: "ব্ব",
    },
    ConjunctRule {
        key: "by",
        value: "ব্য",
    },
    ConjunctRule {
        key: "bY",
        value: "ব্য",
    },
    ConjunctRule {
        key: "br",
        value: "ব্র",
    },
    ConjunctRule {
        key: "bl",
        value: "ব্ল",
    },
    ConjunctRule {
        key: "bhw",
        value: "ভ্ব",
    },
    ConjunctRule {
        key: "vw",
        value: "ভ্ব",
    },
    ConjunctRule {
        key: "bhy",
        value: "ভ্য",
    },
    ConjunctRule {
        key: "bhY",
        value: "ভ্য",
    },
    ConjunctRule {
        key: "vy",
        value: "ভ্য",
    },
    ConjunctRule {
        key: "vY",
        value: "ভ্য",
    },
    ConjunctRule {
        key: "bhr",
        value: "ভ্র",
    },
    ConjunctRule {
        key: "vr",
        value: "ভ্র",
    },
    ConjunctRule {
        key: "bhl",
        value: "ভ্ল",
    },
    ConjunctRule {
        key: "vl",
        value: "ভ্ল",
    },
    ConjunctRule {
        key: "mn",
        value: "ম্ন",
    },
    ConjunctRule {
        key: "mp",
        value: "ম্প",
    },
    ConjunctRule {
        key: "mpr",
        value: "ম্প্র",
    },
    ConjunctRule {
        key: "mf",
        value: "ম্ফ",
    },
    ConjunctRule {
        key: "mph",
        value: "ম্ফ",
    },
    ConjunctRule {
        key: "mw",
        value: "ম্ব",
    },
    ConjunctRule {
        key: "mb",
        value: "ম্ব",
    },
    ConjunctRule {
        key: "mwr",
        value: "ম্ব্র",
    },
    ConjunctRule {
        key: "mbr",
        value: "ম্ব্র",
    },
    ConjunctRule {
        key: "mbh",
        value: "ম্ভ",
    },
    ConjunctRule {
        key: "mv",
        value: "ম্ভ",
    },
    ConjunctRule {
        key: "mbhr",
        value: "ম্ভ্র",
    },
    ConjunctRule {
        key: "mvr",
        value: "ম্ভ্র",
    },
    ConjunctRule {
        key: "mm",
        value: "ম্ম",
    },
    ConjunctRule {
        key: "my",
        value: "ম্য",
    },
    ConjunctRule {
        key: "mY",
        value: "ম্য",
    },
    ConjunctRule {
        key: "mr",
        value: "ম্র",
    },
    ConjunctRule {
        key: "ml",
        value: "ম্ল",
    },
    ConjunctRule {
        key: "yy",
        value: "য্য",
    },
    ConjunctRule {
        key: "zy",
        value: "য্য",
    },
    ConjunctRule {
        key: "zY",
        value: "য্য",
    },
    ConjunctRule {
        key: "rrk",
        value: "র্ক",
    },
    ConjunctRule {
        key: "rrky",
        value: "র্ক্য",
    },
    ConjunctRule {
        key: "rrkY",
        value: "র্ক্য",
    },
    ConjunctRule {
        key: "rrg",
        value: "র্গ",
    },
    ConjunctRule {
        key: "rrgy",
        value: "র্গ্য",
    },
    ConjunctRule {
        key: "rrgY",
        value: "র্গ্য",
    },
    ConjunctRule {
        key: "rrghy",
        value: "র্ঘ্য",
    },
    ConjunctRule {
        key: "rrghY",
        value: "র্ঘ্য",
    },
    ConjunctRule {
        key: "rrNgg",
        value: "র্ঙ্গ",
    },
    ConjunctRule {
        key: "rrcy",
        value: "র্চ্য",
    },
    ConjunctRule {
        key: "rrcY",
        value: "র্চ্য",
    },
    ConjunctRule {
        key: "rrjy",
        value: "র্জ্য",
    },
    ConjunctRule {
        key: "rrjY",
        value: "র্জ্য",
    },
    ConjunctRule {
        key: "rrJy",
        value: "র্জ্য",
    },
    ConjunctRule {
        key: "rrJY",
        value: "র্জ্য",
    },
    ConjunctRule {
        key: "rrjj",
        value: "র্জ্জ",
    },
    ConjunctRule {
        key: "rrjJ",
        value: "র্জ্জ",
    },
    ConjunctRule {
        key: "rrJj",
        value: "র্জ্জ",
    },
    ConjunctRule {
        key: "rrJJ",
        value: "র্জ্জ",
    },
    ConjunctRule {
        key: "rrjNG",
        value: "র্জ্ঞ",
    },
    ConjunctRule {
        key: "rrJNG",
        value: "র্জ্ঞ",
    },
    ConjunctRule {
        key: "rrjn",
        value: "র্জ্ঞ",
    },
    ConjunctRule {
        key: "rrJn",
        value: "র্জ্ঞ",
    },
    ConjunctRule {
        key: "rrNy",
        value: "র্ণ্য",
    },
    ConjunctRule {
        key: "rrNY",
        value: "র্ণ্য",
    },
    ConjunctRule {
        key: "rrty",
        value: "র্ত্য",
    },
    ConjunctRule {
        key: "rrtY",
        value: "র্ত্য",
    },
    ConjunctRule {
        key: "rrthy",
        value: "র্থ্য",
    },
    ConjunctRule {
        key: "rrthY",
        value: "র্থ্য",
    },
    ConjunctRule {
        key: "rrwy",
        value: "র্ব্য",
    },
    ConjunctRule {
        key: "rrwY",
        value: "র্ব্য",
    },
    ConjunctRule {
        key: "rrmy",
        value: "র্ম্য",
    },
    ConjunctRule {
        key: "rrmY",
        value: "র্ম্য",
    },
    ConjunctRule {
        key: "rrshy",
        value: "র্শ্য",
    },
    ConjunctRule {
        key: "rrshY",
        value: "র্শ্য",
    },
    ConjunctRule {
        key: "rrSy",
        value: "র্শ্য",
    },
    ConjunctRule {
        key: "rrSY",
        value: "র্শ্য",
    },
    ConjunctRule {
        key: "rrShy",
        value: "র্ষ্য",
    },
    ConjunctRule {
        key: "rrShY",
        value: "র্ষ্য",
    },
    ConjunctRule {
        key: "rrhy",
        value: "র্হ্য",
    },
    ConjunctRule {
        key: "rrhY",
        value: "র্হ্য",
    },
    ConjunctRule {
        key: "rrkh",
        value: "র্খ",
    },
    ConjunctRule {
        key: "rrgr",
        value: "র্গ্র",
    },
    ConjunctRule {
        key: "rrgh",
        value: "র্ঘ",
    },
    ConjunctRule {
        key: "rrc",
        value: "র্চ",
    },
    ConjunctRule {
        key: "rrch",
        value: "র্ছ",
    },
    ConjunctRule {
        key: "rrj",
        value: "র্জ",
    },
    ConjunctRule {
        key: "rrJ",
        value: "র্জ",
    },
    ConjunctRule {
        key: "rrjh",
        value: "র্ঝ",
    },
    ConjunctRule {
        key: "rrT",
        value: "র্ট",
    },
    ConjunctRule {
        key: "rrD",
        value: "র্ড",
    },
    ConjunctRule {
        key: "rrN",
        value: "র্ণ",
    },
    ConjunctRule {
        key: "rrt",
        value: "র্ত",
    },
    ConjunctRule {
        key: "rrtm",
        value: "র্ত্ম",
    },
    ConjunctRule {
        key: "rrtr",
        value: "র্ত্র",
    },
    ConjunctRule {
        key: "rrth",
        value: "র্থ",
    },
    ConjunctRule {
        key: "rrd",
        value: "র্দ",
    },
    ConjunctRule {
        key: "rrdw",
        value: "র্দ্ব",
    },
    ConjunctRule {
        key: "rrdr",
        value: "র্দ্র",
    },
    ConjunctRule {
        key: "rrdh",
        value: "র্ধ",
    },
    ConjunctRule {
        key: "rrdhw",
        value: "র্ধ্ব",
    },
    ConjunctRule {
        key: "rrn",
        value: "র্ন",
    },
    ConjunctRule {
        key: "rrp",
        value: "র্প",
    },
    ConjunctRule {
        key: "rrf",
        value: "র্ফ",
    },
    ConjunctRule {
        key: "rrph",
        value: "র্ফ",
    },
    ConjunctRule {
        key: "rrw",
        value: "র্ব",
    },
    ConjunctRule {
        key: "rrbh",
        value: "র্ভ",
    },
    ConjunctRule {
        key: "rrv",
        value: "র্ভ",
    },
    ConjunctRule {
        key: "rrm",
        value: "র্ম",
    },
    ConjunctRule {
        key: "rry",
        value: "র্য",
    },
    ConjunctRule {
        key: "rrY",
        value: "র্য",
    },
    ConjunctRule {
        key: "rrl",
        value: "র্ল",
    },
    ConjunctRule {
        key: "rrsh",
        value: "র্শ",
    },
    ConjunctRule {
        key: "rrS",
        value: "র্শ",
    },
    ConjunctRule {
        key: "rrshw",
        value: "র্শ্ব",
    },
    ConjunctRule {
        key: "rrSw",
        value: "র্শ্ব",
    },
    ConjunctRule {
        key: "rrSh",
        value: "র্ষ",
    },
    ConjunctRule {
        key: "rrShT",
        value: "র্ষ্ট",
    },
    ConjunctRule {
        key: "rrShN",
        value: "র্ষ্ণ",
    },
    ConjunctRule {
        key: "rrShNy",
        value: "র্ষ্ণ্য",
    },
    ConjunctRule {
        key: "rrShNY",
        value: "র্ষ্ণ্য",
    },
    ConjunctRule {
        key: "rrs",
        value: "র্স",
    },
    ConjunctRule {
        key: "rrh",
        value: "র্হ",
    },
    ConjunctRule {
        key: "rrDhy",
        value: "র্ঢ্য",
    },
    ConjunctRule {
        key: "rrDhY",
        value: "র্ঢ্য",
    },
    ConjunctRule {
        key: "lk",
        value: "ল্ক",
    },
    ConjunctRule {
        key: "lky",
        value: "ল্ক্য",
    },
    ConjunctRule {
        key: "lkY",
        value: "ল্ক্য",
    },
    ConjunctRule {
        key: "lg",
        value: "ল্গ",
    },
    ConjunctRule {
        key: "lT",
        value: "ল্ট",
    },
    ConjunctRule {
        key: "lD",
        value: "ল্ড",
    },
    ConjunctRule {
        key: "lp",
        value: "ল্প",
    },
    ConjunctRule {
        key: "lf",
        value: "ল্ফ",
    },
    ConjunctRule {
        key: "lph",
        value: "ল্ফ",
    },
    ConjunctRule {
        key: "lw",
        value: "ল্ব",
    },
    ConjunctRule {
        key: "lbh",
        value: "ল্ভ",
    },
    ConjunctRule {
        key: "lv",
        value: "ল্ভ",
    },
    ConjunctRule {
        key: "lm",
        value: "ল্ম",
    },
    ConjunctRule {
        key: "ly",
        value: "ল্য",
    },
    ConjunctRule {
        key: "lY",
        value: "ল্য",
    },
    ConjunctRule {
        key: "ll",
        value: "ল্ল",
    },
    ConjunctRule {
        key: "shc",
        value: "শ্চ",
    },
    ConjunctRule {
        key: "Sc",
        value: "শ্চ",
    },
    ConjunctRule {
        key: "shch",
        value: "শ্ছ",
    },
    ConjunctRule {
        key: "Sch",
        value: "শ্ছ",
    },
    ConjunctRule {
        key: "shn",
        value: "শ্ন",
    },
    ConjunctRule {
        key: "Sn",
        value: "শ্ন",
    },
    ConjunctRule {
        key: "shw",
        value: "শ্ব",
    },
    ConjunctRule {
        key: "Sw",
        value: "শ্ব",
    },
    ConjunctRule {
        key: "shm",
        value: "শ্ম",
    },
    ConjunctRule {
        key: "Sm",
        value: "শ্ম",
    },
    ConjunctRule {
        key: "shy",
        value: "শ্য",
    },
    ConjunctRule {
        key: "shY",
        value: "শ্য",
    },
    ConjunctRule {
        key: "Sy",
        value: "শ্য",
    },
    ConjunctRule {
        key: "SY",
        value: "শ্য",
    },
    ConjunctRule {
        key: "shr",
        value: "শ্র",
    },
    ConjunctRule {
        key: "Sr",
        value: "শ্র",
    },
    ConjunctRule {
        key: "shl",
        value: "শ্ল",
    },
    ConjunctRule {
        key: "Sl",
        value: "শ্ল",
    },
    ConjunctRule {
        key: "Shk",
        value: "ষ্ক",
    },
    ConjunctRule {
        key: "Shkw",
        value: "ষ্ক্ব",
    },
    ConjunctRule {
        key: "Shkr",
        value: "ষ্ক্র",
    },
    ConjunctRule {
        key: "ShT",
        value: "ষ্ট",
    },
    ConjunctRule {
        key: "ShTy",
        value: "ষ্ট্য",
    },
    ConjunctRule {
        key: "ShTY",
        value: "ষ্ট্য",
    },
    ConjunctRule {
        key: "ShTr",
        value: "ষ্ট্র",
    },
    ConjunctRule {
        key: "ShTh",
        value: "ষ্ঠ",
    },
    ConjunctRule {
        key: "ShThy",
        value: "ষ্ঠ্য",
    },
    ConjunctRule {
        key: "ShThY",
        value: "ষ্ঠ্য",
    },
    ConjunctRule {
        key: "ShN",
        value: "ষ্ণ",
    },
    ConjunctRule {
        key: "ShNw",
        value: "ষ্ণ্ব",
    },
    ConjunctRule {
        key: "Shp",
        value: "ষ্প",
    },
    ConjunctRule {
        key: "Shpr",
        value: "ষ্প্র",
    },
    ConjunctRule {
        key: "Shf",
        value: "ষ্ফ",
    },
    ConjunctRule {
        key: "Shph",
        value: "ষ্ফ",
    },
    ConjunctRule {
        key: "Shw",
        value: "ষ্ব",
    },
    ConjunctRule {
        key: "Shm",
        value: "ষ্ম",
    },
    ConjunctRule {
        key: "Shy",
        value: "ষ্য",
    },
    ConjunctRule {
        key: "ShY",
        value: "ষ্য",
    },
    ConjunctRule {
        key: "sk",
        value: "স্ক",
    },
    ConjunctRule {
        key: "skr",
        value: "স্ক্র",
    },
    ConjunctRule {
        key: "skh",
        value: "স্খ",
    },
    ConjunctRule {
        key: "sT",
        value: "স্ট",
    },
    ConjunctRule {
        key: "sTr",
        value: "স্ট্র",
    },
    ConjunctRule {
        key: "st",
        value: "স্ত",
    },
    ConjunctRule {
        key: "stw",
        value: "স্ত্ব",
    },
    ConjunctRule {
        key: "sty",
        value: "স্ত্য",
    },
    ConjunctRule {
        key: "stY",
        value: "স্ত্য",
    },
    ConjunctRule {
        key: "str",
        value: "স্ত্র",
    },
    ConjunctRule {
        key: "sth",
        value: "স্থ",
    },
    ConjunctRule {
        key: "sthy",
        value: "স্থ্য",
    },
    ConjunctRule {
        key: "sthY",
        value: "স্থ্য",
    },
    ConjunctRule {
        key: "sn",
        value: "স্ন",
    },
    ConjunctRule {
        key: "sny",
        value: "স্ন্য",
    },
    ConjunctRule {
        key: "snY",
        value: "স্ন্য",
    },
    ConjunctRule {
        key: "sp",
        value: "স্প",
    },
    ConjunctRule {
        key: "spr",
        value: "স্প্র",
    },
    ConjunctRule {
        key: "spl",
        value: "স্প্‌ল",
    },
    ConjunctRule {
        key: "sf",
        value: "স্ফ",
    },
    ConjunctRule {
        key: "sph",
        value: "স্ফ",
    },
    ConjunctRule {
        key: "sw",
        value: "স্ব",
    },
    ConjunctRule {
        key: "sm",
        value: "স্ম",
    },
    ConjunctRule {
        key: "sy",
        value: "স্য",
    },
    ConjunctRule {
        key: "sY",
        value: "স্য",
    },
    ConjunctRule {
        key: "sr",
        value: "স্র",
    },
    ConjunctRule {
        key: "sl",
        value: "স্ল",
    },
    ConjunctRule {
        key: "hN",
        value: "হ্ণ",
    },
    ConjunctRule {
        key: "hn",
        value: "হ্ন",
    },
    ConjunctRule {
        key: "hw",
        value: "হ্ব",
    },
    ConjunctRule {
        key: "hm",
        value: "হ্ম",
    },
    ConjunctRule {
        key: "hy",
        value: "হ্য",
    },
    ConjunctRule {
        key: "hY",
        value: "হ্য",
    },
    ConjunctRule {
        key: "hr",
        value: "হ্র",
    },
    ConjunctRule {
        key: "hl",
        value: "হ্ল",
    },
];

impl ConjunctDefinitions {
    /// Create a new instance of conjunct definitions
    pub fn new() -> Self {
        // Initialize containers
        let conjunct_trie = ConjunctTrie::with_capacity(conjunct_trie_node_capacity());

        let mut instance = ConjunctDefinitions { conjunct_trie };

        for rule in CONJUNCT_RULES {
            instance.add_conjunct(rule.key, rule.value);
        }

        instance.conjunct_trie.sort_edges();

        // Return the populated instance
        instance
    }

    /// Add a conjunct mapping
    fn add_conjunct(&mut self, key: &'static str, value: &'static str) {
        self.conjunct_trie.insert(key, value);
    }

    /// Check if a sequence can form a valid conjunct
    pub fn can_form_conjunct(&self, key: &str) -> bool {
        self.create_conjunct(key).is_some()
    }

    /// Create a conjunct from a sequence of consonants
    pub fn create_conjunct(&self, key: &str) -> Option<&'static str> {
        let node = self.conjunct_trie.advance(self.conjunct_trie.root(), key)?;
        self.conjunct_trie.value(node)
    }

    fn canonical_conjunct_part<'a>(&self, part: &'a str) -> &'a str {
        canonical_conjunct_part(part)
    }

    /// Create a conjunct from already-tokenized component parts without
    /// allocating a joined key.
    pub fn create_conjunct_from_parts(&self, parts: &[&str]) -> Option<&'static str> {
        if parts.len() < 2 {
            return None;
        }

        let mut node = self.conjunct_trie.root();
        for part in parts {
            node = self
                .conjunct_trie
                .advance(node, self.canonical_conjunct_part(part))?;
        }

        self.conjunct_trie.value(node)
    }

    /// Check component parts without allocating a joined key.
    pub fn can_form_conjunct_from_parts(&self, parts: &[&str]) -> bool {
        self.create_conjunct_from_parts(parts).is_some()
    }

    /// Return the root trie cursor for incremental conjunct matching.
    pub(crate) fn conjunct_match_root(&self) -> usize {
        self.conjunct_trie.root()
    }

    /// Advance an incremental conjunct match by one romanized component.
    pub(crate) fn advance_conjunct_match(&self, node: usize, part: &str) -> Option<usize> {
        self.conjunct_trie
            .advance(node, self.canonical_conjunct_part(part))
    }

    /// Return the conjunct value at an incremental trie cursor, if terminal.
    pub(crate) fn conjunct_match_value(&self, node: usize) -> Option<&'static str> {
        self.conjunct_trie.value(node)
    }

    /// Get romanized consonants for a conjunct
    pub fn get_components(&self, conjunct: &str) -> Option<Vec<String>> {
        for rule in CONJUNCT_RULES {
            if rule.value == conjunct {
                return Some(self.components_for_key(rule.key));
            }
        }
        None
    }

    fn components_for_key(&self, key: &str) -> Vec<String> {
        let mut components = Vec::new();
        let mut i = 0;
        while i < key.len() {
            let mut found = false;
            for len in (1..=key.len() - i).rev() {
                let substr = &key[i..i + len];
                if consonant_value(substr).is_some() || is_special_form_key(substr) {
                    components.push(substr.to_string());
                    i += len;
                    found = true;
                    break;
                }
            }
            if !found {
                components.push(key[i..i + 1].to_string());
                i += 1;
            }
        }
        components
    }

    /// Check if a given sequence is a valid conjunct
    pub fn is_valid_conjunct(&self, components: &[String]) -> bool {
        if components.is_empty() {
            return false;
        }

        let mut node = self.conjunct_trie.root();
        for component in components {
            let Some(next_node) = self
                .conjunct_trie
                .advance(node, self.canonical_conjunct_part(component))
            else {
                return false;
            };
            node = next_node;
        }

        self.conjunct_trie.value(node).is_some()
    }

    /// Get all valid conjuncts
    pub fn get_all_valid_conjuncts(&self) -> BTreeSet<&'static str> {
        CONJUNCT_RULES.iter().map(|rule| rule.key).collect()
    }

    /// Check if a form is a special form (like reph, ya-phola, ba-phola)
    pub fn is_special_form(&self, form: &str) -> bool {
        is_special_form_key(form)
    }
}

impl Default for ConjunctDefinitions {
    fn default() -> Self {
        Self::new()
    }
}

fn canonical_conjunct_part(part: &str) -> &str {
    match part {
        "chh" | "C" | "Ch" | "CH" | "Chh" | "CHH" => "ch",
        "Kh" | "KH" => "kh",
        "Gh" | "GH" => "gh",
        "J" => "j",
        "Jh" | "JH" => "jh",
        "TH" => "Th",
        "DH" => "Dh",
        "Ph" | "PH" | "f" => "ph",
        "Bh" | "BH" | "v" => "bh",
        "Y" => "y",
        "S" => "sh",
        "SH" => "Sh",
        _ => part,
    }
}

fn is_special_form_key(form: &str) -> bool {
    matches!(form, "rr" | "y" | "Y" | "w")
}

/// Return a singleton instance of ConjunctDefinitions
pub fn conjuncts() -> &'static ConjunctDefinitions {
    static INSTANCE: OnceLock<ConjunctDefinitions> = OnceLock::new();
    INSTANCE.get_or_init(ConjunctDefinitions::new)
}

#[derive(Debug)]
struct ConjunctTrie {
    nodes: Vec<ConjunctTrieNode>,
}

#[derive(Debug, Default)]
struct ConjunctTrieNode {
    value: Option<&'static str>,
    edges: Vec<ConjunctTrieEdge>,
}

#[derive(Debug, Clone, Copy)]
struct ConjunctTrieEdge {
    byte: u8,
    node: usize,
}

impl ConjunctTrie {
    fn with_capacity(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(ConjunctTrieNode::default());

        Self { nodes }
    }

    fn root(&self) -> usize {
        0
    }

    fn insert(&mut self, key: &'static str, value: &'static str) {
        let mut node = self.root();

        for byte in key.bytes() {
            node = self.child_or_insert(node, byte);
        }

        assert!(
            self.nodes[node].value.is_none(),
            "duplicate conjunct trie key: {key}"
        );
        self.nodes[node].value = Some(value);
    }

    fn child_or_insert(&mut self, node: usize, byte: u8) -> usize {
        if let Some(edge) = self.nodes[node].edges.iter().find(|edge| edge.byte == byte) {
            return edge.node;
        }

        let child = self.nodes.len();
        self.nodes.push(ConjunctTrieNode::default());
        self.nodes[node]
            .edges
            .push(ConjunctTrieEdge { byte, node: child });
        child
    }

    fn sort_edges(&mut self) {
        for node in &mut self.nodes {
            node.edges.sort_unstable_by_key(|edge| edge.byte);
        }
    }

    fn advance(&self, node: usize, part: &str) -> Option<usize> {
        let mut current = node;

        for byte in part.bytes() {
            current = self.nodes.get(current)?.child(byte)?;
        }

        Some(current)
    }

    fn value(&self, node: usize) -> Option<&'static str> {
        self.nodes.get(node)?.value
    }
}

impl ConjunctTrieNode {
    fn child(&self, byte: u8) -> Option<usize> {
        self.edges
            .binary_search_by_key(&byte, |edge| edge.byte)
            .ok()
            .map(|index| self.edges[index].node)
    }
}

fn conjunct_trie_node_capacity() -> usize {
    1 + CONJUNCT_RULES
        .iter()
        .map(|rule| rule.key.len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjunct_rule_table_has_unique_keys() {
        let mut keys = BTreeSet::new();

        for rule in CONJUNCT_RULES {
            assert!(!rule.key.is_empty());
            assert!(!rule.value.is_empty());
            assert!(
                keys.insert(rule.key),
                "duplicate conjunct rule key: {}",
                rule.key
            );
        }
    }

    #[test]
    fn conjunct_definitions_load_every_static_rule() {
        let definitions = ConjunctDefinitions::new();

        assert_eq!(
            definitions.get_all_valid_conjuncts().len(),
            CONJUNCT_RULES.len()
        );

        for rule in CONJUNCT_RULES {
            assert_eq!(definitions.create_conjunct(rule.key), Some(rule.value));
        }
    }
}
