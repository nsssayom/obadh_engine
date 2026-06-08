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
}
