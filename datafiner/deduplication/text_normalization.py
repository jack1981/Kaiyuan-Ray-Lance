"""Text normalization helpers used by MinHash deduplication.

Normalization aims to improve near-duplicate recall by reducing superficial
differences (case, punctuation, whitespace, diacritics, number forms).
"""

import re
import regex
import unicodedata

PUNCTUATION = (
    "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»•"
    + "".join(
        map(
            chr,
            (
                x
                for a, b in ((0, 9), (11, 13), (13, 32), (127, 160))
                for x in range(a, b)
            ),
        )
    )
)
PUNCTUATION_TRANS = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))
# Match digits in any script, allowing for different decimal separators
# One or more digits in any script
# Common decimal separators (period, comma, Arabic decimal, etc)
# Optional decimal part with digits
# we need regex and not re for this one to match unicode
NUMBERS_PATTERN = regex.compile(
    r"\p{Nd}+([.,،٫⎖⎗⎘]{1}\p{Nd}+)?",
    regex.VERBOSE | regex.UNICODE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def text_normalization(
    text: str,
    lowercase: bool = True,
    norm_whitespace: bool = True,
    remove_punctuation: bool = True,
    norm_unicode_diacritics: bool = True,
    norm_numbers: bool = True,
) -> str:
    """Normalize text for duplicate matching recall.

    Args:
        text: Raw text input.
        lowercase: Whether to lowercase before other transformations.
        norm_whitespace: Whether to collapse whitespace runs.
        remove_punctuation: Whether to translate configured punctuation to space.
        norm_unicode_diacritics: Whether to strip diacritic marks.
        norm_numbers: Whether to normalize unicode digits to `0`.

    Returns:
        Normalized text string.

    Side effects:
        None.

    Assumptions:
        Transformation order is fixed to keep incremental behavior predictable
        across mixed scripts and punctuation styles.
    """
    # NOTE(readability): The order below is intentional; number normalization
    # before punctuation/whitespace cleanup yields more stable signatures.
    if lowercase:
        text = text.lower()
    if norm_numbers:
        text = NUMBERS_PATTERN.sub("0", text)

    # convert punctuation to spaces
    if remove_punctuation:
        text = text.translate(PUNCTUATION_TRANS)

    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    if norm_whitespace:
        text = WHITESPACE_PATTERN.sub(" ", text.strip())
    # diacritics/unicode normalization
    if norm_unicode_diacritics:
        text = "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    return text.strip()
