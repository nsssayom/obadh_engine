use obadh_engine::{ObadhEngine, PhoneticUnitType, Tokenizer};

#[test]
fn test_avro_style_chh_alias() {
    let tokenizer = Tokenizer::new();
    let units = tokenizer.tokenize_word("chhi");

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].text, "chhi");
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("chhi"), "ছি");
    assert_eq!(engine.transliterate("korchhi"), "করছি");
}

#[test]
fn test_chhi_alias_does_not_rewrite_independent_hi_after_r() {
    let tokenizer = Tokenizer::new();
    let units = tokenizer.tokenize_word("rhi");

    assert_eq!(units.len(), 2);
    assert_eq!(units[0].text, "r");
    assert_eq!(units[0].unit_type, PhoneticUnitType::Consonant);
    assert_eq!(units[1].text, "hi");
    assert_eq!(units[1].unit_type, PhoneticUnitType::ConsonantWithVowel);

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("rhi rrhi korchhi"), "রহি র্হি করছি");
}

#[test]
fn test_avro_style_uppercase_aspirated_cha_aliases() {
    let tokenizer = Tokenizer::new();

    for input in ["Ci", "Chi", "Chhi"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    }

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("Cobi Chobi Chhobi"), "ছবি ছবি ছবি");
}

#[test]
fn test_titlecase_aspirated_digraph_aliases_are_composable() {
    let tokenizer = Tokenizer::new();

    for input in ["Kh", "Gh", "Jh", "Ph", "Bh", "KH", "GH", "JH", "PH", "BH"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, PhoneticUnitType::Consonant);
    }

    for input in [
        "Khi", "GhA", "Jhu", "Phe", "BhI", "KHi", "GHee", "JHoo", "JHuu", "PHu", "BHI",
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate(
            "Kh Gh Jh Ph Bh KH GH JH PH BH Khi GhA Jhu Phe BhI KHi GHee JHoo JHuu PHu BHI"
        ),
        "খ ঘ ঝ ফ ভ খ ঘ ঝ ফ ভ খি ঘা ঝু ফে ভী খি ঘী ঝু ঝূ ফু ভী"
    );
}

#[test]
fn test_uppercase_aspirated_aliases_cover_retroflex_cha_and_sha() {
    let tokenizer = Tokenizer::new();

    for input in ["CH", "CHH", "TH", "DH", "SH"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, PhoneticUnitType::Consonant);
    }

    for input in ["CHi", "CHHi", "THa", "DHii", "SHa"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("CH CHH TH DH SH CHi CHHi THa DHii SHa"),
        "ছ ছ ঠ ঢ ষ ছি ছি ঠা ঢী ষা"
    );
}

#[test]
fn test_aspirated_aliases_compose_through_conjunct_canonicalization() {
    let tokenizer = Tokenizer::new();

    for input in [
        "KHy", "GHr", "DHy", "PHr", "BHy", "NTHa", "SHka", "acCHa", "acCHHa",
    ] {
        let units = tokenizer.tokenize_word(input);
        assert!(
            units.iter().any(|unit| {
                matches!(
                    unit.unit_type,
                    PhoneticUnitType::Conjunct
                        | PhoneticUnitType::ConjunctWithVowel
                        | PhoneticUnitType::ConjunctWithTerminator
                )
            }),
            "{input} should compose through canonical conjunct aliases"
        );
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("KHy GHr DHy PHr BHy NTHa SHka acCHa acCHHa ngGHAt"),
        "খ্য ঘ্র ঢ্য ফ্র ভ্য ণ্ঠা ষ্কা আচ্ছা আচ্ছা ঙ্ঘাত"
    );
}

#[test]
fn test_uppercase_cha_aliases_in_cch_conjuncts() {
    let tokenizer = Tokenizer::new();

    for input in ["acCa", "acCha", "acChha"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 2);
        assert_eq!(units[1].unit_type, PhoneticUnitType::ConjunctWithVowel);
    }

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("acCa acCha acChha"), "আচ্ছা আচ্ছা আচ্ছা");
}

#[test]
fn test_pronounced_kkh_alias_for_orthographic_ksh() {
    let tokenizer = Tokenizer::new();

    for input in ["kkh", "kkha", "kkhya", "kkhmI"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert!(
            matches!(
                units[0].unit_type,
                PhoneticUnitType::Conjunct
                    | PhoneticUnitType::ConjunctWithVowel
                    | PhoneticUnitType::ConjunctWithTerminator
            ),
            "{input} should tokenize through the ক্ষ conjunct alias"
        );
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("shikkha kkhom kkhoti okkhor kkhmI"),
        "শিক্ষা ক্ষম ক্ষতি অক্ষর ক্ষ্মী"
    );
}

#[test]
fn test_jn_alias_for_orthographic_jna_conjunct() {
    let tokenizer = Tokenizer::new();

    for input in ["jn", "Jn", "jnan", "Jnaan", "rrjn", "rrJna"] {
        let units = tokenizer.tokenize_word(input);
        assert!(
            units.iter().any(|unit| {
                matches!(
                    unit.unit_type,
                    PhoneticUnitType::Conjunct
                        | PhoneticUnitType::ConjunctWithVowel
                        | PhoneticUnitType::ConjunctWithTerminator
                        | PhoneticUnitType::RephOverConsonant
                        | PhoneticUnitType::RephOverConsonantWithVowel
                )
            }),
            "{input} should tokenize through the জ্ঞ conjunct alias"
        );
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("jNG JNG jn Jn jnan Jnaan bijnaan rrjna"),
        "জ্ঞ জ্ঞ জ্ঞ জ্ঞ জ্ঞান জ্ঞান বিজ্ঞান র্জ্ঞা"
    );
}

#[test]
fn test_external_layout_aliases_are_not_imported_without_obadh_rule_reason() {
    let tokenizer = Tokenizer::new();

    for input in ["q", "Q", "G"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text, input);
        assert_eq!(units[0].unit_type, PhoneticUnitType::Unknown);
    }

    let gg_units = tokenizer.tokenize_word("gg");
    assert_eq!(
        gg_units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("g", PhoneticUnitType::Consonant),
            ("g", PhoneticUnitType::Consonant),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("q qa Q G Ga gg jNG jn"),
        "q qআ Q G Gআ গগ জ্ঞ জ্ঞ"
    );
}

#[test]
fn test_longest_first_vowel_matching_is_deterministic() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("kOI kOU krri"), "কৈ কৌ কৃ");
}

#[test]
fn test_vocalic_r_vowel_wins_over_shorter_reph_signal() {
    let tokenizer = Tokenizer::new();

    let standalone = tokenizer.tokenize_word("rria");
    assert_eq!(
        standalone
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rri", PhoneticUnitType::Vowel),
            ("a", PhoneticUnitType::Vowel)
        ]
    );

    let after_consonant = tokenizer.tokenize_word("krrio");
    assert_eq!(
        after_consonant
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("krri", PhoneticUnitType::ConsonantWithVowel),
            ("o", PhoneticUnitType::TerminatingVowel),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("rria rrio krria krrio"), "ঋআ ঋঅ কৃআ কৃঅ");
    assert_eq!(engine.transliterate("rrhi rrhri"), "র্হি র্হরি");
}

#[test]
fn test_aa_alias_is_single_long_a_vowel() {
    let tokenizer = Tokenizer::new();

    let standalone = tokenizer.tokenize_word("aa");
    assert_eq!(standalone.len(), 1);
    assert_eq!(standalone[0].text, "aa");
    assert_eq!(standalone[0].unit_type, PhoneticUnitType::Vowel);

    let after_consonant = tokenizer.tokenize_word("kaa");
    assert_eq!(after_consonant.len(), 1);
    assert_eq!(after_consonant[0].text, "kaa");
    assert_eq!(
        after_consonant[0].unit_type,
        PhoneticUnitType::ConsonantWithVowel
    );

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("aa A kaa kA"), "আ আ কা কা");
}

#[test]
fn test_doubled_vowel_aliases_are_single_rule_units() {
    let tokenizer = Tokenizer::new();

    for input in ["ee", "ii", "oo", "uu"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should be one vowel unit");
        assert_eq!(units[0].text, input);
        assert_eq!(units[0].unit_type, PhoneticUnitType::Vowel);
    }

    for input in ["kee", "kii", "koo", "kuu"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units.len(),
            1,
            "{input} should be one consonant-with-vowel unit"
        );
        assert_eq!(units[0].text, input);
        assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    }

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("ee ii oo uu"), "ঈ ঈ উ ঊ");
    assert_eq!(engine.transliterate("kee kii koo kuu"), "কী কী কু কূ");
}

#[test]
fn test_documented_vowel_sequences_are_single_rule_units() {
    let tokenizer = Tokenizer::new();

    for input in ["ai", "au", "ae", "ao", "ia", "io", "eo"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should be one vowel unit");
        assert_eq!(units[0].text, input);
        assert_eq!(units[0].unit_type, PhoneticUnitType::Vowel);
    }

    for input in ["kai", "kau", "kae", "kao", "kia", "kio", "keo"] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units.len(),
            1,
            "{input} should be one consonant-with-vowel unit"
        );
        assert_eq!(units[0].text, input);
        assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("ai au ae ao ia io eo"),
        "আই আউ আএ আও ইয়া ইও এও"
    );
    assert_eq!(
        engine.transliterate("kai kau kae kao kia kio keo"),
        "কাই কাউ কাএ কাও কিয়া কিও কেও"
    );
}

#[test]
fn test_uppercase_e_alias_composes_as_e_kar() {
    let tokenizer = Tokenizer::new();

    for (input, expected_text, expected_type) in [
        ("E", "E", PhoneticUnitType::Vowel),
        ("kE", "kE", PhoneticUnitType::ConsonantWithVowel),
        ("kkE", "k,,kE", PhoneticUnitType::ConjunctWithVowel),
        ("rrkE", "rrkE", PhoneticUnitType::RephOverConsonantWithVowel),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should be one composed unit");
        assert_eq!(units[0].text, expected_text);
        assert_eq!(units[0].unit_type, expected_type);
    }

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("E kE kkE rrkE"), "এ কে ক্কে র্কে");
}

#[test]
fn test_vowel_modifier_examples_follow_engine_phonetics() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("ca^d"), "চাঁদ");
    assert_eq!(engine.transliterate("cha^d"), "ছাঁদ");
    assert_eq!(engine.transliterate("du:kh"), "দুঃখ");
}

#[test]
fn test_trailing_diacritics_are_explicit_ordered_marks() {
    let tokenizer = Tokenizer::new();

    let units = tokenizer.tokenize_word("kkA^:");
    assert_eq!(units.len(), 3);
    assert_eq!(units[0].text, "k,,kA");
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConjunctWithVowel);
    assert_eq!(units[1].text, "^");
    assert_eq!(units[1].unit_type, PhoneticUnitType::SpecialForm);
    assert_eq!(units[2].text, ":");
    assert_eq!(units[2].unit_type, PhoneticUnitType::SpecialForm);

    let swapped_units = tokenizer.tokenize_word("ka:^");
    assert_eq!(
        swapped_units
            .iter()
            .map(|unit| unit.text.as_str())
            .collect::<Vec<_>>(),
        vec!["ka", ":", "^"]
    );

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("o^ so^ kA^ kkA^ kko^ rrmo^ rrmA^"),
        "অঁ সঁ কাঁ ক্কাঁ ক্কঁ র্মঁ র্মাঁ"
    );
    assert_eq!(engine.transliterate("ka^: ka:^"), "কাঁঃ কাঃঁ");
}

#[test]
fn test_standalone_diacritic_markers_render_as_rule_signals() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("^ : ^: :^ a ^"), "ঁ ঃ ঁঃ ঃঁ আ ঁ");
}

#[test]
fn test_explicit_hasant_marker_renders_as_a_rule_signal() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate(",, k,, k,,k ,,"), "্ ক্ ক্ক ্");
    assert_eq!(engine.transliterate("kk,, k,,k,,"), "ক্ক্ ক্ক্");
}

#[test]
fn test_vowels_after_dead_or_marked_consonants_render_independently() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("k,,a k,,i k,,I k,,u k,,e k,,O k,,OI k,,OU"),
        "ক্আ ক্ই ক্ঈ ক্উ ক্এ ক্ও ক্ঐ ক্ঔ"
    );
    assert_eq!(
        engine.transliterate("k^a k:a knga t``a rra"),
        "কঁআ কঃআ কংআ ৎআ র্আ"
    );

    assert_eq!(engine.transliterate("ka^ kA^ k,,ka"), "কাঁ কাঁ ক্কা");
}

#[test]
fn test_deliberate_input_sequences_for_orthographic_forms() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("chhOT nArI puruSh bidyut`` bidyuT``"),
        "ছোট নারী পুরুষ বিদ্যুৎ বিদ্যুৎ"
    );
}

#[test]
fn test_mixed_bengali_and_roman_input_is_preserved() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("আমি banglay লিখি। ami banglay gan gai"),
        "আমি বাংলায় লিখি। আমি বাংলায় গান গাই"
    );
}

#[test]
fn test_special_sequences_after_bengali_text_are_byte_safe() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("আমি k,,k t`` T``। তুমি n,,d,,r"),
        "আমি ক্ক ৎ ৎ। তুমি ন্দ্র"
    );
}

#[test]
fn test_visible_a_suppressed_before_conjunct_cluster() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("bhakt"), "ভক্ত");
    assert_eq!(engine.transliterate("shakti"), "শক্তি");
    assert_eq!(
        engine.transliterate("strI bhakt prokash korchhi"),
        "স্ত্রী ভক্ত প্রকাশ করছি"
    );
}

#[test]
fn test_visible_a_kept_in_open_syllables() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("banglay"), "বাংলায়");
    assert_eq!(engine.transliterate("lal"), "লাল");
    assert_eq!(engine.transliterate("kakatuta"), "কাকাতুতা");
}

#[test]
fn test_anusvara_boundary_preserves_following_conjunct_run() {
    let tokenizer = Tokenizer::new();
    let units = tokenizer.tokenize_word("songskrriti");

    assert_eq!(units.len(), 4);
    assert_eq!(units[1].text, "ng");
    assert_eq!(units[1].unit_type, PhoneticUnitType::SpecialForm);
    assert_eq!(units[2].text, "s,,krri");
    assert_eq!(units[2].unit_type, PhoneticUnitType::ConjunctWithVowel);

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("songskrriti songbad songket"),
        "সংস্কৃতি সংবাদ সংকেত"
    );
}

#[test]
fn test_doubled_g_after_ng_is_velar_nasal_conjunct_alias() {
    let tokenizer = Tokenizer::new();
    let units = tokenizer.tokenize_word("bonggo");

    assert_eq!(
        units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("bo", PhoneticUnitType::ConsonantWithTerminator),
            ("Ng,,go", PhoneticUnitType::ConjunctWithTerminator),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("bonggo onggo songgIt ngga ngghAt ngghya ngGhy"),
        "বঙ্গ অঙ্গ সঙ্গীত ঙ্গা ঙ্ঘাত ঙ্ঘ্যা ঙ্ঘ্য"
    );
}

#[test]
fn test_adjacent_normalization_runs_do_not_shift_tail_units() {
    let tokenizer = Tokenizer::new();

    let repeated_reph = tokenizer.tokenize_word("rr,,krr,,ga");
    assert_eq!(
        repeated_reph
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rrk", PhoneticUnitType::RephOverConsonant),
            ("rrga", PhoneticUnitType::RephOverConsonantWithVowel),
        ]
    );

    let repeated_velar = tokenizer.tokenize_word("nggnggha");
    assert_eq!(
        repeated_velar
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("Ng,,g", PhoneticUnitType::Conjunct),
            ("Ng,,gha", PhoneticUnitType::ConjunctWithVowel),
        ]
    );
}

#[test]
fn test_plain_ng_remains_anusvara_before_non_g_velars() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("bangla songbad songket shongkha songskrriti"),
        "বাংলা সংবাদ সংকেত শংখা সংস্কৃতি"
    );
    assert_eq!(engine.transliterate("oNgko oNgk oNggo"), "অঙ্ক অঙ্ক অঙ্গ");
}

#[test]
fn test_ri_phola_words_do_not_collapse_to_vocalic_r() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("kriy kriya"), "ক্রিয় ক্রিয়া");
    assert_eq!(engine.transliterate("prokriya"), "প্রক্রিয়া");
}

#[test]
fn test_iyw_long_iya_signal_is_composable_not_word_final_only() {
    let tokenizer = Tokenizer::new();

    for (input, expected_units, expected_output) in [
        (
            "tiyw",
            vec![
                ("tI", PhoneticUnitType::ConsonantWithVowel),
                ("y", PhoneticUnitType::Consonant),
            ],
            "তীয়",
        ),
        (
            "jatiywta",
            vec![
                ("ja", PhoneticUnitType::ConsonantWithVowel),
                ("tI", PhoneticUnitType::ConsonantWithVowel),
                ("y", PhoneticUnitType::Consonant),
                ("ta", PhoneticUnitType::ConsonantWithVowel),
            ],
            "জাতীয়তা",
        ),
        (
            "ktiYwta",
            vec![
                ("k,,tI", PhoneticUnitType::ConjunctWithVowel),
                ("Y", PhoneticUnitType::Consonant),
                ("ta", PhoneticUnitType::ConsonantWithVowel),
            ],
            "ক্তীয়তা",
        ),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type))
                .collect::<Vec<_>>(),
            expected_units,
            "{input} should tokenize through the reusable iyw signal"
        );

        let engine = ObadhEngine::new();
        assert_eq!(engine.transliterate(input), expected_output);
    }
}

#[test]
fn test_iyw_long_iya_signal_does_not_mutate_vocalic_rri() {
    let tokenizer = Tokenizer::new();

    for (input, expected_units) in [
        (
            "krriyw",
            vec![
                ("krri", PhoneticUnitType::ConsonantWithVowel),
                ("y", PhoneticUnitType::Consonant),
                ("w", PhoneticUnitType::Unknown),
            ],
        ),
        (
            "kiiyw",
            vec![
                ("kii", PhoneticUnitType::ConsonantWithVowel),
                ("y", PhoneticUnitType::Consonant),
                ("w", PhoneticUnitType::Unknown),
            ],
        ),
        (
            "kaiyw",
            vec![
                ("kai", PhoneticUnitType::ConsonantWithVowel),
                ("y", PhoneticUnitType::Consonant),
                ("w", PhoneticUnitType::Unknown),
            ],
        ),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type))
                .collect::<Vec<_>>(),
            expected_units,
            "{input} should not treat a non-short-i vowel as iyw"
        );
    }

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("krriyw kiiyw kaiyw"), "কৃয়w কীয়w কাইয়w");
}

#[test]
fn test_iyw_long_iya_signal_recompacts_following_vowels() {
    let tokenizer = Tokenizer::new();

    for (input, expected_units, expected_output) in [
        (
            "kiywo",
            vec![
                ("kI", PhoneticUnitType::ConsonantWithVowel),
                ("yo", PhoneticUnitType::ConsonantWithTerminator),
            ],
            "কীয়",
        ),
        (
            "kiywe",
            vec![
                ("kI", PhoneticUnitType::ConsonantWithVowel),
                ("ye", PhoneticUnitType::ConsonantWithVowel),
            ],
            "কীয়ে",
        ),
        (
            "kiywO",
            vec![
                ("kI", PhoneticUnitType::ConsonantWithVowel),
                ("yO", PhoneticUnitType::ConsonantWithVowel),
            ],
            "কীয়ো",
        ),
        (
            "jatiywota",
            vec![
                ("ja", PhoneticUnitType::ConsonantWithVowel),
                ("tI", PhoneticUnitType::ConsonantWithVowel),
                ("yo", PhoneticUnitType::ConsonantWithTerminator),
                ("ta", PhoneticUnitType::ConsonantWithVowel),
            ],
            "জাতীয়তা",
        ),
        (
            "jatiywOta",
            vec![
                ("ja", PhoneticUnitType::ConsonantWithVowel),
                ("tI", PhoneticUnitType::ConsonantWithVowel),
                ("yO", PhoneticUnitType::ConsonantWithVowel),
                ("ta", PhoneticUnitType::ConsonantWithVowel),
            ],
            "জাতীয়োতা",
        ),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type))
                .collect::<Vec<_>>(),
            expected_units,
            "{input} should recompact vowels exposed by the iyw marker"
        );

        let engine = ObadhEngine::new();
        assert_eq!(engine.transliterate(input), expected_output);
    }
}

#[test]
fn test_ba_phola_marker_uses_valid_conjunct_table() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("tw dwa biSw kshw kShw kkhw Shkw Shkwa bw bwa"),
        "ত্ব দ্বা বিশ্ব ক্ষ্ব ক্ষ্ব ক্ষ্ব ষ্ক্ব ষ্ক্বা ব্ব ব্বা"
    );
    assert_eq!(
        engine.transliterate("k,,shw k,,shwa r,,rw r,,rwa b,,w b,,wa"),
        "ক্ষ্ব ক্ষ্বা র্ব র্বা ব্ব ব্বা"
    );
    assert_eq!(
        engine.transliterate("rrw rrwa rrwy rrwya rrdw rrdwa rrshw rrshwa rrbw rrbwa"),
        "র্ব র্বা র্ব্য র্ব্যা র্দ্ব র্দ্বা র্শ্ব র্শ্বা র্ব্ব র্ব্বা"
    );
    assert_eq!(engine.transliterate("rry rrya rrY rrYa"), "র্য র্যা র্য র্যা");
    assert_eq!(engine.transliterate("Rw kfw qwa"), "ড়w কফw qwআ");
}

#[test]
fn test_regular_ya_base_accepts_declared_ya_phola_cluster() {
    let tokenizer = Tokenizer::new();

    for (input, expected_text, expected_type) in [
        ("zy", "z,,y", PhoneticUnitType::Conjunct),
        ("zY", "z,,Y", PhoneticUnitType::Conjunct),
        ("zya", "z,,ya", PhoneticUnitType::ConjunctWithVowel),
        ("z,,y", "z,,y", PhoneticUnitType::Conjunct),
        ("z,,Y", "z,,Y", PhoneticUnitType::Conjunct),
        ("z,,ya", "z,,ya", PhoneticUnitType::ConjunctWithVowel),
        ("rrzy", "rr,,z,,y", PhoneticUnitType::Conjunct),
        ("rrzya", "rr,,z,,ya", PhoneticUnitType::ConjunctWithVowel),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should compose as one cluster");
        assert_eq!(units[0].text, expected_text);
        assert_eq!(units[0].unit_type, expected_type);
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("zy zY zya z,,y z,,Y z,,ya rrzy rrzya"),
        "য্য য্য য্যা য্য য্য য্যা র্য্য র্য্যা"
    );
    assert_eq!(engine.transliterate("kz zoy zz"), "কয যয় যয");
}

#[test]
fn test_explicit_hasant_accepts_declared_phola_clusters_only() {
    let tokenizer = Tokenizer::new();

    for (input, expected_text, expected_type) in [
        ("k,,w", "k,,w", PhoneticUnitType::Conjunct),
        ("k,,wa", "k,,wa", PhoneticUnitType::ConjunctWithVowel),
        ("k,,y", "k,,y", PhoneticUnitType::Conjunct),
        ("k,,Ya", "k,,Ya", PhoneticUnitType::ConjunctWithVowel),
        ("S,,w", "S,,w", PhoneticUnitType::Conjunct),
        ("b,,w", "b,,w", PhoneticUnitType::Conjunct),
        ("b,,wa", "b,,wa", PhoneticUnitType::ConjunctWithVowel),
        ("m,,w,,r", "m,,w,,r", PhoneticUnitType::Conjunct),
        (
            "rr,,w,,ya",
            "rr,,w,,ya",
            PhoneticUnitType::ConjunctWithVowel,
        ),
        ("rr,,Y", "rrY", PhoneticUnitType::RephOverConsonant),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units.len(),
            1,
            "{input} should compose as one valid cluster"
        );
        assert_eq!(units[0].text, expected_text);
        assert_eq!(units[0].unit_type, expected_type);
    }

    let invalid = tokenizer.tokenize_word("R,,w");
    assert_eq!(
        invalid
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("R", PhoneticUnitType::Consonant),
            (",,", PhoneticUnitType::ConsonantWithHasant),
            ("w", PhoneticUnitType::Unknown),
        ]
    );

    let invalid = tokenizer.tokenize_word("R,,y");
    assert_eq!(
        invalid
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("R", PhoneticUnitType::Consonant),
            (",,", PhoneticUnitType::ConsonantWithHasant),
            ("y", PhoneticUnitType::Consonant),
        ]
    );

    let invalid = tokenizer.tokenize_word("k,,f,,y");
    assert_eq!(
        invalid
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("k,,f", PhoneticUnitType::Conjunct),
            (",,", PhoneticUnitType::ConsonantWithHasant),
            ("y", PhoneticUnitType::Consonant),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("k,,w k,,wa k,,y k,,Ya S,,w S,,wa b,,w b,,wa m,,w,,r m,,w,,ra rr,,w rr,,w,,ya rr,,Y R,,w R,,y k,,f,,w k,,f,,y"),
        "ক্ব ক্বা ক্য ক্যা শ্ব শ্বা ব্ব ব্বা ম্ব্র ম্ব্রা র্ব র্ব্যা র্য ড়্w ড়্য় ক্ফ্w ক্ফ্য়"
    );
}

#[test]
fn test_reph_ta_and_reph_khanda_ta_are_distinct_rule_paths() {
    let tokenizer = Tokenizer::new();
    let units = tokenizer.tokenize_word("rrt``");

    assert_eq!(
        units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rr", PhoneticUnitType::SpecialForm),
            ("t``", PhoneticUnitType::SpecialForm),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("rrt rrtA rrtm rrtr"), "র্ত র্তা র্ত্ম র্ত্র");
    assert_eq!(
        engine.transliterate("rrt`` rrt``sa rrT`` rrT``sa"),
        "র্ৎ র্ৎসা র্ৎ র্ৎসা"
    );
}

#[test]
fn test_explicit_hasant_after_reph_is_redundant_before_reph_targets() {
    let tokenizer = Tokenizer::new();

    let simple = tokenizer.tokenize_word("rr,,ka");
    assert_eq!(
        simple
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![("rrka", PhoneticUnitType::RephOverConsonantWithVowel)]
    );

    let cluster = tokenizer.tokenize_word("rr,,k,,Sh");
    assert_eq!(
        cluster
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![("rr,,k,,Sh", PhoneticUnitType::Conjunct)]
    );

    let khanda_ta = tokenizer.tokenize_word("rr,,t``sa");
    assert_eq!(
        khanda_ta
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rr", PhoneticUnitType::SpecialForm),
            ("t``", PhoneticUnitType::SpecialForm),
            ("sa", PhoneticUnitType::ConsonantWithVowel),
        ]
    );

    let before_vowel = tokenizer.tokenize_word("rr,,i");
    assert_eq!(
        before_vowel
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rr", PhoneticUnitType::SpecialForm),
            (",,", PhoneticUnitType::ConsonantWithHasant),
            ("i", PhoneticUnitType::Vowel),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("rrk rr,,k rrka rr,,ka rrkSh rr,,k,,Sh rrt``sa rr,,t``sa"),
        "র্ক র্ক র্কা র্কা র্ক্ষ র্ক্ষ র্ৎসা র্ৎসা"
    );
}

#[test]
fn test_explicit_hasant_after_khanda_ta_is_redundant_before_consonants() {
    let tokenizer = Tokenizer::new();

    let standalone = tokenizer.tokenize_word("t``,,sa");
    assert_eq!(
        standalone
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("t``", PhoneticUnitType::SpecialForm),
            ("sa", PhoneticUnitType::ConsonantWithVowel),
        ]
    );

    let with_reph = tokenizer.tokenize_word("rrt``,,sa");
    assert_eq!(
        with_reph
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rr", PhoneticUnitType::SpecialForm),
            ("t``", PhoneticUnitType::SpecialForm),
            ("sa", PhoneticUnitType::ConsonantWithVowel),
        ]
    );

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("t``s t``,,s t``sa t``,,sa rrt``sa rrt``,,sa T``sa rrT``sa"),
        "ৎস ৎস ৎসা ৎসা র্ৎসা র্ৎসা ৎসা র্ৎসা"
    );
}

#[test]
fn test_reph_over_valid_tail_conjuncts_composes_as_one_cluster() {
    let tokenizer = Tokenizer::new();

    for (input, expected_text) in [
        ("rrkSh", "rr,,k,,Sh"),
        ("rrkkh", "rr,,k,,kh"),
        ("rrsk", "rr,,s,,k"),
        ("rrkk", "rr,,k,,k"),
        ("rrzy", "rr,,z,,y"),
        ("rrbw", "rr,,b,,w"),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should compose as one cluster");
        assert_eq!(units[0].text, expected_text);
        assert_eq!(units[0].unit_type, PhoneticUnitType::Conjunct);
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine
            .transliterate("rrkSh rrkSha rrkkh rrkkha rrsk rrska rrkk rrkka rrzy rrzya rrbw rrbwa"),
        "র্ক্ষ র্ক্ষা র্ক্ষ র্ক্ষা র্স্ক র্স্কা র্ক্ক র্ক্কা র্য্য র্য্যা র্ব্ব র্ব্বা"
    );
}

#[test]
fn test_explicit_hasant_reph_tail_clusters_match_implicit_cluster_shape() {
    let tokenizer = Tokenizer::new();

    for (input, expected_text, expected_type) in [
        ("rrk,,Sh", "rr,,k,,Sh", PhoneticUnitType::Conjunct),
        (
            "rrk,,Sha",
            "rr,,k,,Sha",
            PhoneticUnitType::ConjunctWithVowel,
        ),
        ("rrs,,k", "rr,,s,,k", PhoneticUnitType::Conjunct),
        ("rrs,,ka", "rr,,s,,ka", PhoneticUnitType::ConjunctWithVowel),
        ("rrh,,ri", "rr,,h,,ri", PhoneticUnitType::ConjunctWithVowel),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should compose as one cluster");
        assert_eq!(units[0].text, expected_text);
        assert_eq!(units[0].unit_type, expected_type);
    }

    let engine = ObadhEngine::new();
    assert_eq!(
        engine.transliterate("rrkSh rrk,,Sh rrk,,Sha rrsk rrs,,k rrs,,ka rrhri rrh,,ri"),
        "র্ক্ষ র্ক্ষ র্ক্ষা র্স্ক র্স্ক র্স্কা র্হরি র্হ্রি"
    );
}
