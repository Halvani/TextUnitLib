# TextUnitLib/tests/test_textunit_core_basic.py

from __future__ import annotations

import pytest
import spacy

from textunitlib.core import TextUnit


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def tu_en() -> TextUnit:
    """English TextUnit with a simple blank spaCy pipeline."""
    nlp = spacy.blank("en")
    return TextUnit(language=TextUnit.Language.English, nlp=nlp)


# ----------------------------------------------------------------------
# characters()
# ----------------------------------------------------------------------


def test_characters_returns_all_chars_in_order(tu_en: TextUnit):
    text = " Text\nUnit 42 "
    result = tu_en.characters(text)

    # Should be exactly the list of characters from the string
    assert result == list(text)


def test_characters_can_drop_whitespaces(tu_en: TextUnit):
    text = " A\tB \nC "
    result = tu_en.characters(text, drop_whitespaces=True)

    # All whitespace removed, order of remaining chars preserved
    assert result == ["A", "B", "C"]


def test_characters_empty_string(tu_en: TextUnit):
    assert tu_en.characters("") == []


# ----------------------------------------------------------------------
# spaces()
# ----------------------------------------------------------------------


def test_spaces_extracts_only_whitespace(tu_en: TextUnit):
    text = " A\tB \nC"
    result = tu_en.spaces(text)

    # All whitespace characters in order
    assert result == [" ", "\t", " ", "\n"]


def test_spaces_no_whitespace_yields_empty_list(tu_en: TextUnit):
    assert tu_en.spaces("NoSpacesHere") == []


# ----------------------------------------------------------------------
# punctuation_marks()
# ----------------------------------------------------------------------


def test_punctuation_marks_uses_configured_punctuation(tu_en: TextUnit):
    punct_set = tu_en.prop_punctuation
    # If no punctuation resource is available, skip this test
    if len(punct_set) < 1:
        pytest.skip("No punctuation resource loaded.")

    # Take one or two punctuation characters from the configured set
    punct_list = list(punct_set)
    p1 = punct_list[0]
    p2 = punct_list[1] if len(punct_list) > 1 else punct_list[0]

    text = f"A{p1}B{p2}C"
    result = tu_en.punctuation_marks(text)

    # We only inserted these two punctuation chars, so we expect them in order
    assert result == [p1, p2]


def test_punctuation_marks_ignores_non_punctuation(tu_en: TextUnit):
    text = "Hello World"
    result = tu_en.punctuation_marks(text)
    assert result == []


# ----------------------------------------------------------------------
# vowels()
# ----------------------------------------------------------------------


def test_vowels_ignores_consonants(tu_en: TextUnit):
    text = "bcdfg"
    assert tu_en.vowels(text) == []


# ----------------------------------------------------------------------
# letters()
# ----------------------------------------------------------------------


def test_letters_returns_only_alphabetic_chars(tu_en: TextUnit):
    text = "Te2xt-Ü!"
    result = tu_en.letters(text)

    # Only alphabetic characters in order, including Unicode letters
    assert result == ["T", "e", "x", "t", "Ü"]


def test_letters_empty_if_no_alpha(tu_en: TextUnit):
    text = "1234 !!!"
    assert tu_en.letters(text) == []


# ----------------------------------------------------------------------
# digits()
# ----------------------------------------------------------------------


def test_digits_returns_only_digits(tu_en: TextUnit):
    text = "abc123xyz456"
    result = tu_en.digits(text)
    assert result == ["1", "2", "3", "4", "5", "6"]


def test_digits_empty_if_no_digits(tu_en: TextUnit):
    text = "No digits here!"
    assert tu_en.digits(text) == []
