use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug)]
pub(crate) struct CorpusText {
    pub(crate) text: String,
    pub(crate) source_bytes: u64,
    pub(crate) stats: CorpusSourceStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CorpusSourceStats {
    pub(crate) text_inputs: usize,
    pub(crate) html_inputs: usize,
    pub(crate) json_inputs: usize,
    pub(crate) epub_inputs: usize,
    pub(crate) epub_spine_items: usize,
    pub(crate) epub_fallback_inputs: usize,
    pub(crate) epub_fallback_items: usize,
}

impl CorpusSourceStats {
    fn text_input() -> Self {
        Self {
            text_inputs: 1,
            ..Self::default()
        }
    }

    fn html_input() -> Self {
        Self {
            html_inputs: 1,
            ..Self::default()
        }
    }

    fn json_input() -> Self {
        Self {
            json_inputs: 1,
            ..Self::default()
        }
    }

    fn epub_spine(item_count: usize) -> Self {
        Self {
            epub_inputs: 1,
            epub_spine_items: item_count,
            ..Self::default()
        }
    }

    fn epub_fallback(item_count: usize) -> Self {
        Self {
            epub_inputs: 1,
            epub_fallback_inputs: 1,
            epub_fallback_items: item_count,
            ..Self::default()
        }
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.text_inputs += other.text_inputs;
        self.html_inputs += other.html_inputs;
        self.json_inputs += other.json_inputs;
        self.epub_inputs += other.epub_inputs;
        self.epub_spine_items += other.epub_spine_items;
        self.epub_fallback_inputs += other.epub_fallback_inputs;
        self.epub_fallback_items += other.epub_fallback_items;
    }
}

pub(crate) fn expand_corpus_inputs(
    roots: &[PathBuf],
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut inputs = Vec::new();

    for root in roots {
        collect_corpus_inputs(root, &mut inputs)?;
    }

    inputs.sort();
    inputs.dedup();
    if inputs.is_empty() {
        return Err("no corpus input files found".into());
    }
    Ok(inputs)
}

fn collect_corpus_inputs(
    path: &Path,
    inputs: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        inputs.push(path.to_path_buf());
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_corpus_inputs(&path, inputs)?;
        } else if file_type.is_file() && is_supported_directory_corpus_file(&path) {
            inputs.push(path);
        }
    }

    Ok(())
}

fn is_supported_directory_corpus_file(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "epub" | "html" | "htm" | "xhtml" | "json" | "txt" | "text" | "md" | "markdown"
    )
}

pub(crate) fn read_corpus_text(input: &Path) -> Result<CorpusText, Box<dyn std::error::Error>> {
    if is_epub_path(input) {
        return read_epub_text(input);
    }

    let text = fs::read_to_string(input)?;
    if is_json_path(input) {
        return Ok(CorpusText {
            source_bytes: text.len() as u64,
            text: normalize_bangla_text(&json_to_text(&text)?),
            stats: CorpusSourceStats::json_input(),
        });
    }
    if is_htmlish_path(input) {
        return Ok(CorpusText {
            source_bytes: text.len() as u64,
            text: normalize_bangla_text(&htmlish_to_text(&text)),
            stats: CorpusSourceStats::html_input(),
        });
    }

    Ok(CorpusText {
        source_bytes: text.len() as u64,
        text: normalize_bangla_text(&text),
        stats: CorpusSourceStats::text_input(),
    })
}

pub(crate) fn normalize_bangla_text(text: &str) -> String {
    text.nfc().collect()
}

fn read_epub_text(input: &Path) -> Result<CorpusText, Box<dyn std::error::Error>> {
    let source_bytes = fs::metadata(input)?.len();
    let file = File::open(input)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut text = String::new();
    let spine_member_names = epub_spine_text_member_names(&mut archive)?;
    let (member_names, stats) = if spine_member_names.is_empty() {
        let fallback_member_names = epub_text_member_names(&mut archive)?;
        let stats = CorpusSourceStats::epub_fallback(fallback_member_names.len());
        (fallback_member_names, stats)
    } else {
        let stats = CorpusSourceStats::epub_spine(spine_member_names.len());
        (spine_member_names, stats)
    };

    for name in member_names {
        if let Some(raw) = read_zip_text_member(&mut archive, &name)? {
            if is_plain_text_member(&name) {
                text.push_str(&raw);
            } else {
                text.push_str(&htmlish_to_text(&raw));
            }
            text.push('\n');
        }
    }

    Ok(CorpusText {
        text: normalize_bangla_text(&text),
        source_bytes,
        stats,
    })
}

#[derive(Debug, Clone)]
struct EpubManifestItem {
    href: String,
    properties: String,
}

fn epub_spine_text_member_names(
    archive: &mut zip::ZipArchive<File>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let Some(container) = read_zip_text_member(archive, "META-INF/container.xml")? else {
        return Ok(Vec::new());
    };
    let Some(opf_path) = container_rootfile_path(&container) else {
        return Ok(Vec::new());
    };
    let Some(opf) = read_zip_text_member(archive, &opf_path)? else {
        return Ok(Vec::new());
    };

    Ok(opf_spine_text_member_names(&opf, &opf_path))
}

fn epub_text_member_names(
    archive: &mut zip::ZipArchive<File>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut names = Vec::new();

    for index in 0..archive.len() {
        let member = archive.by_index(index)?;
        let name = member.name().to_string();
        if !member.is_dir() && is_epub_text_member(&name) {
            names.push(name);
        }
    }

    Ok(names)
}

fn read_zip_text_member(
    archive: &mut zip::ZipArchive<File>,
    name: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut member = match archive.by_name(name) {
        Ok(member) => member,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut bytes = Vec::new();
    member.read_to_end(&mut bytes)?;
    Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
}

fn container_rootfile_path(container_xml: &str) -> Option<String> {
    xml_tags_named(container_xml, "rootfile")
        .into_iter()
        .find_map(|tag| xml_attr_value(tag, "full-path"))
        .map(|path| normalize_epub_path(&path))
}

fn opf_spine_text_member_names(opf: &str, opf_path: &str) -> Vec<String> {
    let manifest = opf_manifest(opf, opf_path);
    let mut names = Vec::new();

    for tag in xml_tags_named(opf, "itemref") {
        if xml_attr_value(tag, "linear").is_some_and(|linear| linear.eq_ignore_ascii_case("no")) {
            continue;
        }
        let Some(idref) = xml_attr_value(tag, "idref") else {
            continue;
        };
        let Some(item) = manifest.get(&idref) else {
            continue;
        };
        if item
            .properties
            .split_ascii_whitespace()
            .any(|property| property.eq_ignore_ascii_case("nav"))
        {
            continue;
        }
        names.push(item.href.clone());
    }

    deduplicate_preserving_order(names)
}

fn opf_manifest(opf: &str, opf_path: &str) -> BTreeMap<String, EpubManifestItem> {
    let mut manifest = BTreeMap::new();

    for tag in xml_tags_named(opf, "item") {
        let Some(id) = xml_attr_value(tag, "id") else {
            continue;
        };
        let Some(href) = xml_attr_value(tag, "href") else {
            continue;
        };
        let media_type = xml_attr_value(tag, "media-type").unwrap_or_default();
        if !is_epub_text_media_type(&media_type) && !is_epub_text_member(&href) {
            continue;
        }
        manifest.insert(
            id,
            EpubManifestItem {
                href: resolve_epub_href(opf_path, &href),
                properties: xml_attr_value(tag, "properties").unwrap_or_default(),
            },
        );
    }

    manifest
}

fn is_epub_text_media_type(media_type: &str) -> bool {
    matches!(
        media_type.to_ascii_lowercase().as_str(),
        "application/xhtml+xml" | "text/html" | "text/plain"
    )
}

fn resolve_epub_href(opf_path: &str, href: &str) -> String {
    let href = href.split('#').next().unwrap_or(href);
    let combined = if href.starts_with('/') {
        href.trim_start_matches('/').to_string()
    } else if let Some((base, _)) = opf_path.rsplit_once('/') {
        format!("{base}/{href}")
    } else {
        href.to_string()
    };
    normalize_epub_path(&combined)
}

fn normalize_epub_path(path: &str) -> String {
    let mut parts = Vec::new();

    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }

    parts.join("/")
}

fn deduplicate_preserving_order(values: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();

    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }

    unique
}

fn is_epub_path(input: &Path) -> bool {
    input
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("epub"))
}

fn is_htmlish_path(input: &Path) -> bool {
    input
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "html" | "htm" | "xhtml"
            )
        })
}

fn is_json_path(input: &Path) -> bool {
    input
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

fn is_epub_text_member(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    matches!(
        Path::new(&name)
            .extension()
            .and_then(|extension| extension.to_str()),
        Some("xhtml" | "html" | "htm" | "txt")
    )
}

fn is_plain_text_member(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("txt"))
}

fn json_to_text(input: &str) -> Result<String, serde_json::Error> {
    let value = serde_json::from_str::<Value>(input)?;
    let mut output = String::new();
    append_json_prose_text(&value, &mut output);
    Ok(output)
}

fn append_json_prose_text(value: &Value, output: &mut String) {
    match value {
        Value::Object(object) => {
            let mut emitted_known_field = false;
            for key in ["title", "headline", "content", "text", "body", "article"] {
                if let Some(value) = object.get(key) {
                    append_json_prose_text(value, output);
                    emitted_known_field = true;
                }
            }
            if emitted_known_field {
                return;
            }
            for value in object.values() {
                append_json_prose_text(value, output);
            }
        }
        Value::Array(values) => {
            for value in values {
                append_json_prose_text(value, output);
            }
        }
        Value::String(text) => {
            output.push_str(text);
            output.push('\n');
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn xml_tags_named<'a>(xml: &'a str, expected_name: &str) -> Vec<&'a str> {
    let mut tags = Vec::new();
    let mut cursor = 0;

    while let Some(start) = xml[cursor..].find('<') {
        let tag_start = cursor + start + 1;
        let Some(end) = xml[tag_start..].find('>') else {
            break;
        };
        let tag_end = tag_start + end;
        let tag = &xml[tag_start..tag_end];
        if xml_tag_has_name(tag, expected_name) {
            tags.push(tag);
        }
        cursor = tag_end + 1;
    }

    tags
}

fn xml_tag_has_name(tag: &str, expected_name: &str) -> bool {
    let tag = tag.trim_start();
    if tag.starts_with('/') || tag.starts_with('?') || tag.starts_with('!') {
        return false;
    }

    let name = tag
        .split(|ch: char| ch.is_ascii_whitespace() || ch == '/')
        .next()
        .unwrap_or_default();
    xml_local_name(name).eq_ignore_ascii_case(expected_name)
}

fn xml_attr_value(tag: &str, expected_name: &str) -> Option<String> {
    let mut index = 0;

    while index < tag.len() {
        index = skip_xml_space_and_slash(tag, index);
        let name_start = index;
        while let Some(ch) = char_at(tag, index) {
            if !is_xml_attr_name_char(ch) {
                break;
            }
            index += ch.len_utf8();
        }
        if name_start == index {
            index = advance_char(tag, index)?;
            continue;
        }

        let name = &tag[name_start..index];
        index = skip_xml_space(tag, index);
        if char_at(tag, index) != Some('=') {
            continue;
        }
        index += 1;
        index = skip_xml_space(tag, index);
        let quote = char_at(tag, index)?;
        if quote != '"' && quote != '\'' {
            continue;
        }
        index += 1;
        let value_start = index;
        while let Some(ch) = char_at(tag, index) {
            if ch == quote {
                let value = &tag[value_start..index];
                index += ch.len_utf8();
                if xml_local_name(name).eq_ignore_ascii_case(expected_name) {
                    return Some(decode_entities_to_string(value));
                }
                break;
            }
            index += ch.len_utf8();
        }
    }

    None
}

fn xml_local_name(name: &str) -> &str {
    name.rsplit_once(':')
        .map(|(_, local)| local)
        .unwrap_or(name)
}

fn is_xml_attr_name_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.')
}

fn skip_xml_space(text: &str, mut index: usize) -> usize {
    while let Some(ch) = char_at(text, index) {
        if !ch.is_ascii_whitespace() {
            break;
        }
        index += ch.len_utf8();
    }
    index
}

fn skip_xml_space_and_slash(text: &str, mut index: usize) -> usize {
    while let Some(ch) = char_at(text, index) {
        if !ch.is_ascii_whitespace() && ch != '/' {
            break;
        }
        index += ch.len_utf8();
    }
    index
}

fn char_at(text: &str, index: usize) -> Option<char> {
    text.get(index..)?.chars().next()
}

fn advance_char(text: &str, index: usize) -> Option<usize> {
    Some(index + char_at(text, index)?.len_utf8())
}

fn htmlish_to_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    let mut skip_until: Option<&'static str> = None;

    while index < input.len() {
        if input[index..].starts_with('<') {
            let Some(tag_end_offset) = input[index + 1..].find('>') else {
                break;
            };
            let tag = &input[index + 1..index + 1 + tag_end_offset];
            if let Some(expected_tag) = skip_until {
                if html_tag_closes(tag, expected_tag) {
                    skip_until = None;
                    output.push(' ');
                }
            } else {
                if let Some(skip_tag) = html_opening_skip_tag(tag) {
                    skip_until = Some(skip_tag);
                }
                output.push(' ');
            }
            index += tag_end_offset + 2;
            continue;
        }

        let Some(ch) = char_at(input, index) else {
            break;
        };
        if skip_until.is_some() {
            index += ch.len_utf8();
            continue;
        }

        if ch == '&' {
            if let Some((decoded, next_index)) = decode_html_entity_at(input, index + 1) {
                output.push(decoded);
                index = next_index;
            } else {
                output.push(' ');
                index += ch.len_utf8();
            }
        } else {
            output.push(ch);
            index += ch.len_utf8();
        }
    }

    output
}

fn html_opening_skip_tag(tag: &str) -> Option<&'static str> {
    let tag = tag.trim_start();
    if tag.starts_with('/') || tag.starts_with('!') || tag.starts_with('?') {
        return None;
    }
    match html_tag_name(tag).to_ascii_lowercase().as_str() {
        "script" => Some("script"),
        "style" => Some("style"),
        _ => None,
    }
}

fn html_tag_closes(tag: &str, expected_name: &str) -> bool {
    let Some(tag) = tag.trim_start().strip_prefix('/') else {
        return false;
    };
    html_tag_name(tag).eq_ignore_ascii_case(expected_name)
}

fn html_tag_name(tag: &str) -> &str {
    let name = tag
        .trim_start()
        .split(|ch: char| ch.is_ascii_whitespace() || ch == '/')
        .next()
        .unwrap_or_default();
    xml_local_name(name)
}

fn decode_html_entity_at(input: &str, entity_start: usize) -> Option<(char, usize)> {
    let rest = input.get(entity_start..)?;
    let semicolon = rest.find(';')?;
    if semicolon > 32 {
        return None;
    }
    let entity = &rest[..semicolon];
    let decoded = decode_html_entity_name(entity).unwrap_or(' ');
    Some((decoded, entity_start + semicolon + 1))
}

fn decode_entities_to_string(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '&' {
            output.push(decode_html_entity(&mut chars).unwrap_or(' '));
        } else {
            output.push(ch);
        }
    }

    output
}

fn decode_html_entity<I>(chars: &mut std::iter::Peekable<I>) -> Option<char>
where
    I: Iterator<Item = char>,
{
    let mut entity = String::new();
    while let Some(ch) = chars.next() {
        if ch == ';' {
            return decode_html_entity_name(&entity);
        }
        entity.push(ch);
        if entity.len() > 32 {
            return None;
        }
    }
    None
}

fn decode_html_entity_name(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        _ => entity
            .strip_prefix("#x")
            .or_else(|| entity.strip_prefix("#X"))
            .and_then(|hex| u32::from_str_radix(hex, 16).ok())
            .or_else(|| {
                entity
                    .strip_prefix('#')
                    .and_then(|decimal| decimal.parse::<u32>().ok())
            })
            .and_then(char::from_u32),
    }
}

pub(crate) struct BanglaTokenIter<'a> {
    text: &'a str,
    chars: std::str::CharIndices<'a>,
    start: Option<usize>,
    end: usize,
}

impl<'a> BanglaTokenIter<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        Self {
            text,
            chars: text.char_indices(),
            start: None,
            end: 0,
        }
    }
}

impl<'a> Iterator for BanglaTokenIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.chars.next() {
                Some((index, ch)) if is_bangla_token_char(ch) => {
                    if self.start.is_none() {
                        self.start = Some(index);
                    }
                    self.end = index + ch.len_utf8();
                }
                Some(_) => {
                    if let Some(start) = self.start.take() {
                        let token = &self.text[start..self.end];
                        if is_bangla_lexicon_word(token) {
                            return Some(token);
                        }
                    }
                }
                None => {
                    if let Some(start) = self.start.take() {
                        let token = &self.text[start..self.end];
                        if is_bangla_lexicon_word(token) {
                            return Some(token);
                        }
                    }
                    return None;
                }
            }
        }
    }
}

fn is_bangla_token_char(ch: char) -> bool {
    is_bengali_block_word_char(ch) || is_joiner(ch)
}

pub(crate) fn is_bangla_lexicon_word(word: &str) -> bool {
    let mut has_base = false;
    let mut previous_joiner = false;
    let mut previous_hasant = false;

    for (index, ch) in word.chars().enumerate() {
        if is_joiner(ch) {
            if index == 0 || previous_joiner {
                return false;
            }
            previous_joiner = true;
            continue;
        }

        if !is_bangla_word_char(ch) {
            return false;
        }
        if !has_base && !is_bangla_base_char(ch) {
            return false;
        }
        if previous_hasant && !is_bangla_base_char(ch) {
            return false;
        }
        if previous_hasant && ch == '\u{09CD}' {
            return false;
        }

        previous_joiner = false;
        previous_hasant = ch == '\u{09CD}';
        has_base |= is_bangla_base_char(ch);
    }

    has_base && !previous_joiner && !previous_hasant
}

pub(crate) fn is_clean_roman_word_input(word: &str) -> bool {
    let mut has_ascii_letter = false;

    for ch in word.chars() {
        if ch.is_ascii_alphabetic() {
            has_ascii_letter = true;
            continue;
        }
        if matches!(ch, ',' | '`' | '/') {
            continue;
        }
        return false;
    }

    has_ascii_letter
}

fn is_bangla_word_char(ch: char) -> bool {
    is_bangla_base_char(ch)
        || matches!(
            ch,
            '\u{0981}'..='\u{0983}'
                | '\u{09BC}'
                | '\u{09BE}'..='\u{09C4}'
                | '\u{09C7}'..='\u{09C8}'
                | '\u{09CB}'..='\u{09CD}'
                | '\u{09D7}'
                | '\u{09E2}'..='\u{09E3}'
                | '\u{09FE}'
        )
}

fn is_bengali_block_word_char(ch: char) -> bool {
    is_bangla_word_char(ch) || matches!(ch, '\u{09F0}'..='\u{09F1}')
}

fn is_bangla_base_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{0985}'..='\u{098C}'
            | '\u{098F}'..='\u{0990}'
            | '\u{0993}'..='\u{09A8}'
            | '\u{09AA}'..='\u{09B0}'
            | '\u{09B2}'
            | '\u{09B6}'..='\u{09B9}'
            | '\u{09CE}'
            | '\u{09DC}'..='\u{09DD}'
            | '\u{09DF}'..='\u{09E1}'
    )
}

fn is_joiner(ch: char) -> bool {
    matches!(ch, '\u{200C}' | '\u{200D}')
}
