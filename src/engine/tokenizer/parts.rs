const MAX_INLINE_PARTS: usize = 8;

pub(super) struct BorrowedParts<'a> {
    inline: [&'a str; MAX_INLINE_PARTS],
    len: usize,
    overflow: Option<Vec<&'a str>>,
}

impl<'a> BorrowedParts<'a> {
    pub(super) fn new() -> Self {
        Self {
            inline: [""; MAX_INLINE_PARTS],
            len: 0,
            overflow: None,
        }
    }

    pub(super) fn from_one(first: &'a str) -> Self {
        let mut parts = Self::new();
        parts.push(first);
        parts
    }

    pub(super) fn from_two(first: &'a str, second: &'a str) -> Self {
        let mut parts = Self::new();
        parts.push(first);
        parts.push(second);
        parts
    }

    #[inline]
    pub(super) fn push(&mut self, part: &'a str) {
        if let Some(parts) = &mut self.overflow {
            parts.push(part);
            return;
        }

        if self.len < MAX_INLINE_PARTS {
            self.inline[self.len] = part;
            self.len += 1;
            return;
        }

        let mut parts = Vec::with_capacity(MAX_INLINE_PARTS * 2);
        parts.extend_from_slice(&self.inline[..self.len]);
        parts.push(part);
        self.overflow = Some(parts);
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.overflow.as_ref().map_or(self.len, std::vec::Vec::len)
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[&'a str] {
        self.overflow.as_deref().unwrap_or(&self.inline[..self.len])
    }

    pub(super) fn extended_is_valid(
        parts: &[&str],
        next: &str,
        conjunct_defs: &crate::definitions::conjuncts::ConjunctDefinitions,
    ) -> bool {
        if parts.len() < MAX_INLINE_PARTS {
            let mut borrowed = [""; MAX_INLINE_PARTS];
            borrowed[..parts.len()].copy_from_slice(parts);
            borrowed[parts.len()] = next;
            return conjunct_defs.can_form_conjunct_from_parts(&borrowed[..parts.len() + 1]);
        }

        let mut borrowed = Vec::with_capacity(parts.len() + 1);
        borrowed.extend_from_slice(parts);
        borrowed.push(next);
        conjunct_defs.can_form_conjunct_from_parts(borrowed.as_slice())
    }
}
