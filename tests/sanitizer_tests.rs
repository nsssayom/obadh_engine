use obadh_engine::Sanitizer;

#[test]
fn test_valid_input() {
    let sanitizer = Sanitizer::new();

    // Valid inputs
    assert!(sanitizer.sanitize("Hello World").is_ok());
    assert!(sanitizer.sanitize("abc123").is_ok());
    assert!(sanitizer.sanitize("k,, kaaj").is_ok()); // Explicit hasant marker
    assert!(sanitizer.sanitize("123").is_ok()); // Numerals
    assert!(sanitizer.sanitize("!@#$%^&*()").is_ok()); // Special characters
    assert!(sanitizer.sanitize("আমি banglay লিখি।").is_ok()); // Mixed Bengali/Roman input
    assert!(sanitizer.sanitize("ami\nbangla\tlekhi\r\n").is_ok()); // Structural whitespace
    assert!(sanitizer.sanitize("ami\u{00a0}bangla\u{2003}lekhi").is_ok());
}

#[test]
fn test_invalid_input() {
    let sanitizer = Sanitizer::new();

    // Invalid inputs containing unsupported non-Latin characters
    assert!(sanitizer.sanitize("こんにちは").is_err()); // Japanese
    assert!(sanitizer.sanitize("Привет").is_err()); // Russian
}

#[test]
fn test_invalid_character_error_order_is_deterministic() {
    let sanitizer = Sanitizer::new();

    assert_eq!(
        sanitizer.sanitize("x😀é😀").unwrap_err(),
        "Invalid characters found: é😀"
    );
}

#[test]
fn test_custom_allowed_characters_extend_default_contract() {
    let sanitizer = Sanitizer::new().with_allowed_chars(&['é']);

    assert!(sanitizer.sanitize("café").is_ok());
    assert_eq!(sanitizer.clean("café😀"), "café");
    assert!(sanitizer.sanitize("café😀").is_err());
}

#[test]
fn test_clean_input() {
    let sanitizer = Sanitizer::new();

    // Clean should preserve Bengali and remove invalid characters
    assert_eq!(sanitizer.clean("Hello অ World"), "Hello অ World");
    assert_eq!(sanitizer.clean("abc123こんにちは"), "abc123");
    assert_eq!(sanitizer.clean("!@#$%^&*()Привет"), "!@#$%^&*()");
    assert_eq!(
        sanitizer.clean("ami\u{00a0}bangla😀\u{2003}lekhi"),
        "ami\u{00a0}bangla\u{2003}lekhi"
    );
}

#[test]
fn test_is_valid() {
    let sanitizer = Sanitizer::new();

    // Test valid and invalid inputs
    assert!(sanitizer.is_valid("Hello World"));
    assert!(sanitizer.is_valid("abc123"));
    assert!(sanitizer.is_valid("অআই"));
    assert!(sanitizer.is_valid("Hello অ World"));
    assert!(sanitizer.is_valid("ami\nbangla\tlekhi\r\n"));
    assert!(sanitizer.is_valid("ami\u{00a0}bangla\u{2003}lekhi"));
    assert!(!sanitizer.is_valid("こんにちは"));
}
