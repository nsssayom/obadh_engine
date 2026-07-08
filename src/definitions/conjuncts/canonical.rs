use crate::definitions::consonant_value;

pub(super) fn canonical_conjunct_part(part: &str) -> &str {
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

/// Roman markers that request a ya-phola (য-ফলা).
pub(super) fn is_ya_phola_marker(part: &str) -> bool {
    matches!(part, "y" | "Y")
}

/// Whether a ya-phola can subjoin directly onto `base_last`, the consonant it
/// would attach to.
///
/// Ya-phola is productive: it composes onto any real consonant base — a single
/// consonant (খ্য, প্য) or the tail of a conjunct (প্ল + য-ফলা = প্ল্য, used by
/// loanwords such as প্ল্যান). Two narrow exceptions are preserved:
///  - র/ড়/ঢ়/ঙ (`r`/`R`/`Rh`/`Ng`) refuse ya-phola and instead yield the standalone
///    forms রয়া/ড়য়া/ঢ়য়া/ঙয়া;
///  - a base that already ends in a phola marker takes no further ya-phola, so
///    শ্ব + য stays শ্বয় and the `iyw` long-ঈয় signal is untouched.
pub(super) fn ya_phola_attaches_to(base_last: &str) -> bool {
    let canonical = canonical_conjunct_part(base_last);
    consonant_value(canonical).is_some()
        && !is_ya_phola_refusing(canonical)
        && !is_phola_marker_component(base_last)
}

fn is_ya_phola_refusing(canonical: &str) -> bool {
    matches!(canonical, "r" | "R" | "Rh" | "Ng")
}

fn is_phola_marker_component(part: &str) -> bool {
    // `y`/`Y` (য়) and `w` (ওয়/ব-ফলা) carry consonant values but are phola markers,
    // not phola bases: a base already ending in one takes no further ya-phola, so
    // শ্ব + য stays শ্বয় rather than শ্ব্য.
    matches!(canonical_conjunct_part(part), "y" | "w")
}

pub(super) fn is_special_form_key(form: &str) -> bool {
    matches!(form, "rr" | "y" | "Y" | "w")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_conjunct_parts_are_declared_aliases_only() {
        let cases = [
            ("chh", "ch"),
            ("CHH", "ch"),
            ("KH", "kh"),
            ("J", "j"),
            ("PH", "ph"),
            ("v", "bh"),
            ("Y", "y"),
            ("S", "sh"),
            ("SH", "Sh"),
            ("q", "q"),
        ];

        for (input, expected) in cases {
            assert_eq!(canonical_conjunct_part(input), expected);
        }
    }

    #[test]
    fn special_conjunct_forms_are_narrow() {
        for form in ["rr", "y", "Y", "w"] {
            assert!(is_special_form_key(form));
        }

        for form in ["z", "b", "Z", "ph"] {
            assert!(!is_special_form_key(form));
        }
    }

    #[test]
    fn ya_phola_attaches_to_real_consonants_except_the_refusing_set() {
        // Any real consonant base takes ya-phola productively: aspirated stops
        // (খ্য/ছ্য/ফ্য), plain stops, sibilants, and the tail of a conjunct (প্ল + য).
        for base_last in ["kh", "Ch", "JH", "TH", "f", "v", "p", "l", "sh", "z", "s"] {
            assert!(ya_phola_attaches_to(base_last), "{base_last:?} should take ya-phola");
        }

        // র/ড়/ঢ়/ঙ refuse ya-phola (রয়া/ড়য়া/ঢ়য়া/ঙয়া); phola markers are not bases;
        // non-consonants (`q`) never take it.
        for base_last in ["r", "R", "Rh", "Ng", "y", "Y", "w", "q"] {
            assert!(!ya_phola_attaches_to(base_last), "{base_last:?} must refuse ya-phola");
        }
    }
}
