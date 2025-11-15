from enum import Enum

import pytest
import spacy

from textunitlib.nlp_handler import (
    NlpHandler,
    SpacyModelSize,
    _norm_language,
    _resolve_model_id,
)


class DummyLanguage(Enum):
    English = 1
    German = 2


# ----------------------------------------------------------------------
# Tests for _norm_language
# ----------------------------------------------------------------------


def test_norm_language_none_defaults_to_english():
    assert _norm_language(None) == "english"


def test_norm_language_accepts_language_like_strings():
    assert _norm_language("English") == "english"
    assert _norm_language("EN") == "english"
    assert _norm_language("german") == "german"
    assert _norm_language("de") == "german"


def test_norm_language_accepts_enum_like_language():
    assert _norm_language(DummyLanguage.English) == "english"
    assert _norm_language(DummyLanguage.German) == "german"


def test_norm_language_raises_on_unsupported_language():
    with pytest.raises(ValueError):
        _norm_language("french")


# ----------------------------------------------------------------------
# Tests for _resolve_model_id
# ----------------------------------------------------------------------


def test_resolve_model_id_with_enum_matching_language():
    model_name = _resolve_model_id("english", SpacyModelSize.English_Medium)
    assert model_name == SpacyModelSize.English_Medium.model_name()


def test_resolve_model_id_raises_on_language_mismatch():
    with pytest.raises(ValueError):
        _resolve_model_id("english", SpacyModelSize.German_Small)


def test_resolve_model_id_accepts_string_model_id():
    name = _resolve_model_id("english", "en_core_web_md")
    assert name == "en_core_web_md"


def test_resolve_model_id_defaults_to_small_models():
    assert _resolve_model_id("english", None) == SpacyModelSize.English_Small.model_name()
    assert _resolve_model_id("german", None) == SpacyModelSize.German_Small.model_name()


def test_resolve_model_id_raises_on_unknown_language():
    with pytest.raises(ValueError):
        _resolve_model_id("french", None)


# ----------------------------------------------------------------------
# Tests for NlpHandler with override pipeline
# ----------------------------------------------------------------------


def test_nlphandler_uses_override_pipeline_and_adds_sentencizer():
    # Create a blank English pipeline without any pipes
    nlp_override = spacy.blank("en")
    assert "sentencizer" not in nlp_override.pipe_names

    handler = NlpHandler(
        language="english",
        nlp=nlp_override,
        ensure_sentencizer=True,
    )

    nlp_obj = handler.get_nlp()

    # Should be exactly the override object
    assert nlp_obj is nlp_override

    # Sentencizer should have been added
    assert "sentencizer" in nlp_obj.pipe_names


def test_sents_uses_sentence_boundaries():
    nlp_override = spacy.blank("en")
    handler = NlpHandler(
        language="english",
        nlp=nlp_override,
        ensure_sentencizer=True,
    )

    text = "This is the first sentence. This is the second one."
    sents = handler.sents(text)

    # Rule based sentencizer should split into two sentences
    assert len(sents) == 2
    assert sents[0].strip().startswith("This is the first")
    assert sents[1].strip().startswith("This is the second")


def test_tokenize_uses_same_nlp_instance_cached():
    nlp_override = spacy.blank("en")
    handler = NlpHandler(
        language="english",
        nlp=nlp_override,
        ensure_sentencizer=False,
    )

    tokens1 = handler.tokenize("One two")
    tokens2 = handler.tokenize("Three four")

    # Same override object used internally
    assert handler.get_nlp() is nlp_override
    assert [t.text for t in tokens1] == ["One", "two"]
    assert [t.text for t in tokens2] == ["Three", "four"]


# ----------------------------------------------------------------------
# Tests for spaCy loading and component pruning via monkeypatch
# ----------------------------------------------------------------------


def test_nlphandler_prunes_components(monkeypatch):
    # Create a fake spaCy pipeline with two named components
    nlp_fake = spacy.blank("en")
    nlp_fake.add_pipe("sentencizer", name="sentencizer")
    nlp_fake.add_pipe("tagger", name="tagger")

    def fake_load(model_id):
        # Ensure we are called with a model name
        assert isinstance(model_id, str)
        return nlp_fake

    monkeypatch.setattr("textunitlib.nlp_handler.spacy.load", fake_load)

    handler = NlpHandler(
        language="english",
        model_id=SpacyModelSize.English_Small,
        use_components=("tagger",),  # keep only tagger
        ensure_sentencizer=False,
        verbose=False,
    )

    nlp_obj = handler.get_nlp()

    # Only tagger should remain
    assert "tagger" in nlp_obj.pipe_names
    assert "sentencizer" not in nlp_obj.pipe_names


def test_nlphandler_adds_sentencizer_if_no_sentence_component(monkeypatch):
    nlp_fake = spacy.blank("en")

    def fake_load(model_id):
        return nlp_fake

    monkeypatch.setattr("textunitlib.nlp_handler.spacy.load", fake_load)

    handler = NlpHandler(
        language="english",
        model_id=SpacyModelSize.English_Small,
        ensure_sentencizer=True,
    )

    nlp_obj = handler.get_nlp()
    assert "sentencizer" in nlp_obj.pipe_names



@pytest.mark.slow
def test_small_english_model_end_to_end():
    handler = NlpHandler(
        language="english",
        model_id=SpacyModelSize.English_Small,
        ensure_sentencizer=True,
        verbose=False,
    )

    nlp = handler.get_nlp()
    doc = nlp("This is a short English sentence.")

    # Basic sanity checks
    assert len(doc) > 0
    # Should have sentence boundaries
    sents = list(doc.sents)
    assert len(sents) >= 1
    # POS tags should be non-empty strings
    assert all(isinstance(t.pos_, str) for t in doc)


@pytest.mark.slow
def test_small_german_model_end_to_end():
    handler = NlpHandler(
        language="german",
        model_id=SpacyModelSize.German_Small,
        ensure_sentencizer=True,
        verbose=False,
    )

    nlp = handler.get_nlp()
    doc = nlp("Dies ist ein kurzer deutscher Beispielsatz.")

    # Basic sanity checks
    assert len(doc) > 0
    sents = list(doc.sents)
    assert len(sents) >= 1
    assert all(isinstance(t.pos_, str) for t in doc)


@pytest.mark.slow
def test_small_english_model_pos_tags_reasonable():
    handler = NlpHandler(
        language="english",
        model_id=SpacyModelSize.English_Small,
        ensure_sentencizer=True,
        verbose=False,
    )

    text = "The dog runs quickly."
    pairs = handler.pos_tags(text)
    token2pos = {tok: pos for tok, pos in pairs}

    # Basic tokens must be there
    assert {"The", "dog", "runs", "quickly", "."}.issubset(token2pos.keys())

    # Reasonable POS expectations, but tolerant to small changes
    assert token2pos["The"] in {"DET", "PRON"}
    assert token2pos["dog"] == "NOUN"
    assert token2pos["runs"] in {"VERB", "AUX"}
    assert token2pos["quickly"] == "ADV"


@pytest.mark.slow
def test_small_german_model_pos_tags_reasonable():
    handler = NlpHandler(
        language="german",
        model_id=SpacyModelSize.German_Small,
        ensure_sentencizer=True,
        verbose=False,
    )

    text = "Der Hund läuft schnell."
    pairs = handler.pos_tags(text)
    token2pos = {tok: pos for tok, pos in pairs}

    # Basic tokens must be there
    assert {"Der", "Hund", "läuft", "schnell", "."}.issubset(token2pos.keys())

    # Reasonable POS expectations, again a bit tolerant
    assert token2pos["Der"] in {"DET", "PRON"}
    assert token2pos["Hund"] == "NOUN"
    assert token2pos["läuft"] == "VERB"
    assert token2pos["schnell"] == "ADV"
