use crate::definitions::conjuncts::ConjunctDefinitions;
use crate::engine::inline_parts::DefaultInlineParts;

pub(super) type BorrowedParts<'a> = DefaultInlineParts<'a>;

pub(super) fn extended_is_valid<'a>(
    parts: &[&'a str],
    next: &'a str,
    conjunct_defs: &ConjunctDefinitions,
) -> bool {
    let mut extended = BorrowedParts::from_slice(parts);
    extended.push(next);
    conjunct_defs.can_form_conjunct_from_parts(extended.as_slice())
}
