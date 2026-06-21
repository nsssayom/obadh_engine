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

/// Productive aspirated-base ya-phola forms are intentionally derived instead
/// of duplicated in the source-owned conjunct table. This covers spellings such
/// as ছ্য/ঝ্য/ফ্য while preserving non-phola base paths like রয়া/ড়য়া.
pub(super) fn is_derived_aspirated_ya_phola(parts: &[&str]) -> bool {
    parts.len() == 2
        && is_aspirated_varga_consonant(parts[0])
        && canonical_conjunct_part(parts[1]) == "y"
}

pub(super) fn is_derived_aspirated_ya_phola_prefix(parts: &[&str]) -> bool {
    match parts {
        [base] => is_aspirated_varga_consonant(base),
        [base, phola] => {
            is_aspirated_varga_consonant(base) && canonical_conjunct_part(phola) == "y"
        }
        _ => false,
    }
}

fn is_aspirated_varga_consonant(part: &str) -> bool {
    matches!(
        canonical_conjunct_part(part),
        "kh" | "gh" | "ch" | "jh" | "Th" | "Dh" | "th" | "dh" | "ph" | "bh"
    )
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
    fn aspirated_ya_phola_is_derived_for_varga_consonants_only() {
        for parts in [
            ["kh", "y"],
            ["Ch", "Y"],
            ["JH", "y"],
            ["TH", "Y"],
            ["f", "y"],
            ["v", "Y"],
        ] {
            assert!(is_derived_aspirated_ya_phola(&parts), "{parts:?}");
        }

        for parts in [
            ["r", "y"],
            ["R", "y"],
            ["Rh", "y"],
            ["Ng", "y"],
            ["sh", "y"],
            ["z", "y"],
        ] {
            assert!(!is_derived_aspirated_ya_phola(&parts), "{parts:?}");
        }
    }
}
