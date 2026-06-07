//! Definitions for Bengali consonants
//!
//! This file contains the mappings for Bengali consonants, organized by their phonetic categories,
//! as well as information about conjunct formation.

use super::insert_unique;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A Roman input rule and its Bengali consonant output.
pub type ConsonantRule = (&'static str, &'static str);

/// Organizes consonants by their phonetic groups (vargas) and characteristics
pub struct ConsonantSystem {
    /// Velar consonants (k-varga)
    pub velars: &'static [ConsonantRule],
    /// Palatal consonants (c-varga)
    pub palatals: &'static [ConsonantRule],
    /// Retroflex consonants (ṭ-varga)
    pub retroflexes: &'static [ConsonantRule],
    /// Dental consonants (t-varga)
    pub dentals: &'static [ConsonantRule],
    /// Labial consonants (p-varga)
    pub labials: &'static [ConsonantRule],
    /// Semivowels and liquids
    pub semivowels: &'static [ConsonantRule],
    /// Fricatives and others
    pub fricatives: &'static [ConsonantRule],
    /// Special consonants
    pub special: &'static [ConsonantRule],
}

const VELARS: &[ConsonantRule] = &[
    ("k", "ক"),  // ka
    ("kh", "খ"), // kha
    ("Kh", "খ"), // titlecase aspirated kha alias
    ("KH", "খ"), // uppercase aspirated kha alias
    ("g", "গ"),  // ga
    ("gh", "ঘ"), // gha
    ("Gh", "ঘ"), // titlecase aspirated gha alias
    ("GH", "ঘ"), // uppercase aspirated gha alias
    ("Ng", "ঙ"), // nga
];

const PALATALS: &[ConsonantRule] = &[
    ("c", "চ"),   // ca
    ("ch", "ছ"),  // cha
    ("chh", "ছ"), // explicit aspirated cha alias
    ("C", "ছ"),   // compact uppercase aspirated cha alias
    ("Ch", "ছ"),  // titlecase aspirated cha alias
    ("CH", "ছ"),  // uppercase aspirated cha alias
    ("Chh", "ছ"), // titlecase aspirated cha alias
    ("CHH", "ছ"), // uppercase aspirated cha alias
    ("J", "জ"),   // ja
    ("j", "জ"),   // ja
    ("jh", "ঝ"),  // jha
    ("Jh", "ঝ"),  // titlecase aspirated jha alias
    ("JH", "ঝ"),  // uppercase aspirated jha alias
    ("NG", "ঞ"),  // nya (palatized)
];

const RETROFLEXES: &[ConsonantRule] = &[
    ("T", "ট"),  // Ta
    ("Th", "ঠ"), // Tha
    ("TH", "ঠ"), // uppercase Tha alias
    ("D", "ড"),  // Da
    ("Dh", "ঢ"), // Dha
    ("DH", "ঢ"), // uppercase Dha alias
    ("N", "ণ"),  // Na
];

const DENTALS: &[ConsonantRule] = &[
    ("t", "ত"),  // ta
    ("th", "থ"), // tha
    ("d", "দ"),  // da
    ("dh", "ধ"), // dha
    ("n", "ন"),  // na (non-palatized)
];

const LABIALS: &[ConsonantRule] = &[
    ("p", "প"),  // pa
    ("ph", "ফ"), // pha
    ("Ph", "ফ"), // titlecase aspirated pha alias
    ("PH", "ফ"), // uppercase aspirated pha alias
    ("f", "ফ"),  // alternative for pha
    ("b", "ব"),  // ba
    ("bh", "ভ"), // bha
    ("Bh", "ভ"), // titlecase aspirated bha alias
    ("BH", "ভ"), // uppercase aspirated bha alias
    ("v", "ভ"),  // alternative for bha
    ("m", "ম"),  // ma
];

const SEMIVOWELS: &[ConsonantRule] = &[
    ("z", "য"), // yô
    ("r", "র"), // rô
    ("l", "ল"), // lô
];

const FRICATIVES: &[ConsonantRule] = &[
    ("sh", "শ"), // palatal śô
    ("S", "শ"),  // palatal śô
    ("Sh", "ষ"), // retroflex ṣô
    ("SH", "ষ"), // uppercase retroflex ṣô alias
    ("s", "স"),  // dental sô
    ("h", "হ"),  // hô
];

const SPECIAL: &[ConsonantRule] = &[
    ("R", "ড়"),  // ṛô
    ("Rh", "ঢ়"), // ṛhô
    ("y", "য়"),  // antastô yô
    ("Y", "য়"),  // antastô yô
];

const CONSONANT_CATEGORIES: [&[ConsonantRule]; 8] = [
    VELARS,
    PALATALS,
    RETROFLEXES,
    DENTALS,
    LABIALS,
    SEMIVOWELS,
    FRICATIVES,
    SPECIAL,
];

const CONSONANT_RULE_COUNT: usize = VELARS.len()
    + PALATALS.len()
    + RETROFLEXES.len()
    + DENTALS.len()
    + LABIALS.len()
    + SEMIVOWELS.len()
    + FRICATIVES.len()
    + SPECIAL.len();

/// Returns a structured system of Bengali consonants
pub const fn consonant_system() -> ConsonantSystem {
    ConsonantSystem {
        velars: VELARS,
        palatals: PALATALS,
        retroflexes: RETROFLEXES,
        dentals: DENTALS,
        labials: LABIALS,
        semivowels: SEMIVOWELS,
        fricatives: FRICATIVES,
        special: SPECIAL,
    }
}

/// Returns ordered consonant rule categories for deterministic matcher construction.
pub const fn consonant_categories() -> [&'static [ConsonantRule]; 8] {
    CONSONANT_CATEGORIES
}

/// Look up a Bengali consonant from a Roman rule key without constructing or hashing a map.
pub fn consonant_value(roman: &str) -> Option<&'static str> {
    match roman {
        "k" => Some("ক"),
        "kh" | "Kh" | "KH" => Some("খ"),
        "g" => Some("গ"),
        "gh" | "Gh" | "GH" => Some("ঘ"),
        "Ng" => Some("ঙ"),
        "c" => Some("চ"),
        "ch" | "chh" | "C" | "Ch" | "CH" | "Chh" | "CHH" => Some("ছ"),
        "J" | "j" => Some("জ"),
        "jh" | "Jh" | "JH" => Some("ঝ"),
        "NG" => Some("ঞ"),
        "T" => Some("ট"),
        "Th" | "TH" => Some("ঠ"),
        "D" => Some("ড"),
        "Dh" | "DH" => Some("ঢ"),
        "N" => Some("ণ"),
        "t" => Some("ত"),
        "th" => Some("থ"),
        "d" => Some("দ"),
        "dh" => Some("ধ"),
        "n" => Some("ন"),
        "p" => Some("প"),
        "ph" | "Ph" | "PH" | "f" => Some("ফ"),
        "b" => Some("ব"),
        "bh" | "Bh" | "BH" | "v" => Some("ভ"),
        "m" => Some("ম"),
        "z" => Some("য"),
        "r" => Some("র"),
        "l" => Some("ল"),
        "sh" | "S" => Some("শ"),
        "Sh" | "SH" => Some("ষ"),
        "s" => Some("স"),
        "h" => Some("হ"),
        "R" => Some("ড়"),
        "Rh" => Some("ঢ়"),
        "y" | "Y" => Some("য়"),
        _ => None,
    }
}

/// Return whether a Roman rule key is a known consonant.
pub fn is_consonant(roman: &str) -> bool {
    consonant_value(roman).is_some()
}

fn build_consonants() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::with_capacity(CONSONANT_RULE_COUNT);

    for category in CONSONANT_CATEGORIES {
        for &(roman, bengali) in category {
            insert_unique(&mut map, "consonant", roman, bengali);
        }
    }

    map
}

/// Returns a shared flattened map of all Bengali consonants.
pub fn consonants_static() -> &'static HashMap<&'static str, &'static str> {
    static INSTANCE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    INSTANCE.get_or_init(build_consonants)
}

/// Returns a flattened map of all Bengali consonants
pub fn consonants() -> HashMap<&'static str, &'static str> {
    consonants_static().clone()
}
