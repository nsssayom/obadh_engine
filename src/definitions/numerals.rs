//! Definitions for Bengali numerals
//!
//! This file contains mappings for Bengali numerals (০-৯).

use super::insert_unique;
use std::collections::HashMap;

/// Return the Bengali numeral for an ASCII decimal digit.
pub fn bengali_digit(digit: char) -> Option<&'static str> {
    match digit {
        '0' => Some("\u{09e6}"),
        '1' => Some("\u{09e7}"),
        '2' => Some("\u{09e8}"),
        '3' => Some("\u{09e9}"),
        '4' => Some("\u{09ea}"),
        '5' => Some("\u{09eb}"),
        '6' => Some("\u{09ec}"),
        '7' => Some("\u{09ed}"),
        '8' => Some("\u{09ee}"),
        '9' => Some("\u{09ef}"),
        _ => None,
    }
}

/// Returns a map of Latin numerals to Bengali numerals
pub fn numerals() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    // Map Latin digits to Bengali digits.
    insert_unique(&mut map, "numeral", "0", "\u{09e6}");
    insert_unique(&mut map, "numeral", "1", "\u{09e7}");
    insert_unique(&mut map, "numeral", "2", "\u{09e8}");
    insert_unique(&mut map, "numeral", "3", "\u{09e9}");
    insert_unique(&mut map, "numeral", "4", "\u{09ea}");
    insert_unique(&mut map, "numeral", "5", "\u{09eb}");
    insert_unique(&mut map, "numeral", "6", "\u{09ec}");
    insert_unique(&mut map, "numeral", "7", "\u{09ed}");
    insert_unique(&mut map, "numeral", "8", "\u{09ee}");
    insert_unique(&mut map, "numeral", "9", "\u{09ef}");

    map
}
