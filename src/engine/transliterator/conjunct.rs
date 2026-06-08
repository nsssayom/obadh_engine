use std::borrow::Cow;

use crate::definitions::{consonant_value, diacritics::HASANT};

use super::Transliterator;

impl Transliterator {
    pub(super) fn append_reph_prefix(output: &mut String) {
        output.push('র');
        output.push_str(HASANT);
    }

    fn conjunct_component(&self, part: &str) -> Option<&'static str> {
        match part {
            "rZ" => Some("র\u{200C}"),
            "y" | "Y" => Some("য"),
            "w" => Some("ব"),
            _ => consonant_value(part),
        }
    }

    pub(super) fn render_conjunct_parts(&self, parts: &[&str]) -> Option<Cow<'static, str>> {
        if parts.len() < 2 {
            return None;
        }

        if let Some(mapped) = self.conjuncts.create_conjunct_from_parts(parts) {
            return Some(Cow::Borrowed(mapped));
        }

        if parts.first() == Some(&"rr") {
            let tail = self.render_conjunct_parts(&parts[1..])?;
            let mut rendered = String::with_capacity("র".len() + HASANT.len() + tail.len());
            Self::append_reph_prefix(&mut rendered);
            rendered.push_str(tail.as_ref());
            return Some(Cow::Owned(rendered));
        }

        let mut rendered = String::new();

        for (index, part) in parts.iter().enumerate() {
            rendered.push_str(self.conjunct_component(part)?);
            if index < parts.len() - 1 {
                rendered.push_str(HASANT);
            }
        }

        Some(Cow::Owned(rendered))
    }
}
