"""Streaming tokenizer.

Rules:
- Text is NFC-normalized, then casefolded ("Straße" -> "strasse").
- A token is a run of letters and digits (any script, via \\w minus "_").
- Apostrophes INSIDE a word are kept: "don't", "п'ять".
  Typographic variants (’ and ʼ) are folded to ASCII "'" so they match.
  Apostrophes at the edges are dropped: "'hello'" -> "hello".
- Hyphens split words: "well-known" -> "well", "known".
- Digits are tokens: "3.12" -> "3", "12" (the dot is punctuation).
"""
import re
import unicodedata
from collections.abc import Iterator

TOKEN_RE = re.compile(r"[^\W_]+(?:['’ʼ][^\W_]+)*")


def tokenize(text: str) -> Iterator[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(text):
        yield match.group().replace("’", "'").replace("ʼ", "'")
