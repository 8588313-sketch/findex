import inspect

from findex.tokenize import tokenize


def toks(text):
    return list(tokenize(text))


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)


def test_mixed_case():
    assert toks("Hello WORLD") == ["hello", "world"]


def test_cyrillic():
    assert toks("Привіт, світе!") == ["привіт", "світе"]


def test_combining_accent():
    assert toks("cafe\u0301") == ["caf\u00e9"]


def test_casefold_eszett():
    assert toks("Straße") == ["strasse"]


def test_apostrophes():
    assert toks("Don't 'hello' п’ять") == ["don't", "hello", "п'ять"]


def test_hyphen_and_digits():
    assert toks("well-known Python 3.12") == ["well", "known", "python", "3", "12"]


def test_empty_and_punctuation():
    assert toks("") == []
    assert toks("...!?") == []
