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
    """Performs the following operations to increase recall when looking for matches between documents:                                - number normalization
    - weekday normalization                                                                                                            - month normalization
    - lowercase text
    - replace all whitespace with a single " "
    - remove all punctuation                                                                                                           - convert diacritics                                                                                                               - unicode normalize
    Args:
        text
                                                                                                                                       Returns:
        modified text
    """
    # We should apply the transformation in such order so that, we do same transformations
    # incrementaly as we would do if we applied each from scratch.                                                                     # Eg.
    # 1|2|3 -> 000
    # vs                                                                                                                               # 1|2|3 -> 0                                                                                                                       lowercase: bool = True
    # lower case
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
