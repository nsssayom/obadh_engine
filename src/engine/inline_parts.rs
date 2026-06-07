const DEFAULT_INLINE_PARTS: usize = 8;

pub(crate) type DefaultInlineParts<'a> = InlineParts<'a, DEFAULT_INLINE_PARTS>;

pub(crate) struct InlineParts<'a, const N: usize = DEFAULT_INLINE_PARTS> {
    inline: [&'a str; N],
    len: usize,
    overflow: Option<Vec<&'a str>>,
}

impl<'a, const N: usize> InlineParts<'a, N> {
    pub(crate) fn new() -> Self {
        Self {
            inline: [""; N],
            len: 0,
            overflow: None,
        }
    }

    pub(crate) fn from_one(first: &'a str) -> Self {
        let mut parts = Self::new();
        parts.push(first);
        parts
    }

    pub(crate) fn from_two(first: &'a str, second: &'a str) -> Self {
        let mut parts = Self::new();
        parts.push(first);
        parts.push(second);
        parts
    }

    pub(crate) fn from_slice(parts: &[&'a str]) -> Self {
        let mut inline_parts = Self::new();

        for part in parts {
            inline_parts.push(part);
        }

        inline_parts
    }

    pub(crate) fn from_text(text: &'a str) -> Self {
        let mut parts = Self::new();

        for part in text.split(",,") {
            parts.push(part);
        }

        parts
    }

    #[inline]
    pub(crate) fn push(&mut self, part: &'a str) {
        if let Some(parts) = &mut self.overflow {
            parts.push(part);
            return;
        }

        if self.len < N {
            self.inline[self.len] = part;
            self.len += 1;
            return;
        }

        let mut parts = Vec::with_capacity(N * 2);
        parts.extend_from_slice(&self.inline[..self.len]);
        parts.push(part);
        self.overflow = Some(parts);
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.overflow.as_ref().map_or(self.len, std::vec::Vec::len)
    }

    #[inline]
    pub(crate) fn last(&self) -> Option<&'a str> {
        self.as_slice().last().copied()
    }

    #[inline]
    pub(crate) fn replace_last(&mut self, part: &'a str) {
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
    pub(crate) fn as_slice(&self) -> &[&'a str] {
        self.overflow.as_deref().unwrap_or(&self.inline[..self.len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_parts_keep_common_clusters_inline() {
        let mut parts = DefaultInlineParts::from_text("rr,,k,,Sh");

        assert!(parts.overflow.is_none());
        assert_eq!(parts.as_slice(), &["rr", "k", "Sh"]);

        parts.replace_last("ShA");
        assert_eq!(parts.as_slice(), &["rr", "k", "ShA"]);
    }

    #[test]
    fn inline_parts_spill_only_for_long_explicit_chains() {
        let mut parts = DefaultInlineParts::from_text("a,,b,,c,,d,,e,,f,,g,,h,,i");

        assert!(parts.overflow.is_some());
        assert_eq!(parts.len(), DEFAULT_INLINE_PARTS + 1);
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

    #[test]
    fn inline_parts_can_extend_borrowed_slices_without_allocating_common_cases() {
        let mut parts = DefaultInlineParts::from_slice(&["k", "Sh"]);

        parts.push("y");

        assert!(parts.overflow.is_none());
        assert_eq!(parts.as_slice(), &["k", "Sh", "y"]);
    }
}
