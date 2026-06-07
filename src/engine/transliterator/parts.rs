const MAX_INLINE_CONJUNCT_PARTS: usize = 8;

pub(super) struct ConjunctParts<'a> {
    inline: [&'a str; MAX_INLINE_CONJUNCT_PARTS],
    len: usize,
    overflow: Option<Vec<&'a str>>,
}

impl<'a> ConjunctParts<'a> {
    #[inline]
    pub(super) fn from_text(text: &'a str) -> Self {
        let mut parts = Self {
            inline: [""; MAX_INLINE_CONJUNCT_PARTS],
            len: 0,
            overflow: None,
        };

        for part in text.split(",,") {
            parts.push(part);
        }

        parts
    }

    #[inline]
    fn push(&mut self, part: &'a str) {
        if let Some(parts) = &mut self.overflow {
            parts.push(part);
            return;
        }

        if self.len < MAX_INLINE_CONJUNCT_PARTS {
            self.inline[self.len] = part;
            self.len += 1;
            return;
        }

        let mut parts = Vec::with_capacity(MAX_INLINE_CONJUNCT_PARTS * 2);
        parts.extend_from_slice(&self.inline[..self.len]);
        parts.push(part);
        self.overflow = Some(parts);
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.overflow.as_ref().map_or(self.len, std::vec::Vec::len)
    }

    #[inline]
    pub(super) fn last(&self) -> Option<&'a str> {
        self.as_slice().last().copied()
    }

    #[inline]
    pub(super) fn replace_last(&mut self, part: &'a str) {
        if let Some(parts) = &mut self.overflow {
            if let Some(last) = parts.last_mut() {
                *last = part;
            }
            return;
        }

        if self.len > 0 {
            self.inline[self.len - 1] = part;
        }
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[&'a str] {
        self.overflow.as_deref().unwrap_or(&self.inline[..self.len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjunct_parts_keep_common_clusters_inline() {
        let mut parts = ConjunctParts::from_text("rr,,k,,Sh");

        assert!(parts.overflow.is_none());
        assert_eq!(parts.as_slice(), &["rr", "k", "Sh"]);

        parts.replace_last("ShA");
        assert_eq!(parts.as_slice(), &["rr", "k", "ShA"]);
    }

    #[test]
    fn conjunct_parts_spill_only_for_long_explicit_chains() {
        let mut parts = ConjunctParts::from_text("a,,b,,c,,d,,e,,f,,g,,h,,i");

        assert!(parts.overflow.is_some());
        assert_eq!(parts.len(), MAX_INLINE_CONJUNCT_PARTS + 1);
        assert_eq!(
            parts.as_slice(),
            &["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        );

        parts.replace_last("z");
        assert_eq!(
            parts.as_slice(),
            &["a", "b", "c", "d", "e", "f", "g", "h", "z"]
        );
    }
}
