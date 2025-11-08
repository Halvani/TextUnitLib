from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Callable, List, Tuple, Iterator, Dict, Any, Union
from functools import cache
import spacy


# ------------------------- Unified model catalog -------------------------

class SpacyModelSize(Enum):
    """
    Unified spaCy model catalog for English and German.

    English:
        Small   -> "en_core_web_sm"
        Medium  -> "en_core_web_md"
        Large   -> "en_core_web_lg"
        Neural  -> "en_core_web_trf"

    German:
        Small   -> "de_core_news_sm"
        Medium  -> "de_core_news_md"
        Large   -> "de_core_news_lg"
        Neural  -> "de_dep_news_trf"
    """
    English_Small = ("english", "en_core_web_sm")
    English_Medium = ("english", "en_core_web_md")
    English_Large = ("english", "en_core_web_lg")
    English_Neural = ("english", "en_core_web_trf")

    German_Small = ("german", "de_core_news_sm")
    German_Medium = ("german", "de_core_news_md")
    German_Large = ("german", "de_core_news_lg")
    German_Neural = ("german", "de_dep_news_trf")

    def language(self) -> str:
        return self.value[0]

    def model_name(self) -> str:
        return self.value[1]


_SUPPORTED_LANG_ALIASES = {
    "english": {"english", "en", "eng"},
    "german": {"german", "de", "deu", "ger"},
}


def _norm_language(lang: Optional[Any]) -> str:
    """
    Normalize a user-provided language hint to 'english' or 'german'.
    Accepts:
      • Your Language enum (Language.English / Language.German)
      • Strings like 'English', 'German', 'en', 'de'
      • None -> 'english'
    """
    if lang is None:
        return "english"

    # Enum case
    if isinstance(lang, Enum):
        candidate = lang.name.lower()
    else:
        candidate = str(lang).lower()

    for key, aliases in _SUPPORTED_LANG_ALIASES.items():
        if candidate in aliases:
            return key

    # Common fallback
    simple = candidate.replace("_", "")
    if simple in {"english", "german"}:
        return simple

    raise ValueError(f"Unsupported language hint '{lang}'. Supported: English, German.")


def _resolve_model_id(
    language: str,
    model_id: Optional[Union[str, SpacyModelSize]]
) -> str:
    """
    Resolve the correct spaCy model ID.
    """
    if isinstance(model_id, SpacyModelSize):
        # Validate that the chosen model matches the requested language
        if model_id.language() != language:
            raise ValueError(
                f"Model {model_id.name} is for {model_id.language()}, "
                f"but language={language} was requested."
            )
        return model_id.model_name()

    if isinstance(model_id, str):
        return model_id

    # Default small models
    if language == "english":
        return SpacyModelSize.English_Small.model_name()
    elif language == "german":
        return SpacyModelSize.German_Small.model_name()
    else:
        raise ValueError(f"Unsupported language '{language}'.")


# ------------------------- NlpHandler -------------------------

class NlpHandler:
    """
    A lightweight, reusable wrapper around a spaCy pipeline that provides:

      • Reliable model loading (with optional auto-download)
      • Simple logging hook (print or custom logger)
      • Tokenization, sentence splitting, POS/lemma utilities
      • Batch processing (nlp.pipe)

    Multilingual note:
      - Pass `language=Language.English` or `Language.German` (your enum)
        or a string like "English"/"German"/"en"/"de".
      - Optionally specify `model_id=SpacyModelSize.English_Large`, etc.
    """

    def __init__(
        self,
        model_id: Optional[Union[str, SpacyModelSize]] = None,
        *,
        language: Optional[Any] = "english",   # ✅ Explicit default
        use_components: Optional[Iterable[str]] = ("tagger", "senter", "lemmatizer", "morphologizer", "attribute_ruler"),
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
        nlp: Optional["spacy.language.Language"] = None,
        project_name: str = "[NLP]"
    ):
        self.language = _norm_language(language)
        self.model_id = _resolve_model_id(self.language, model_id)
        self.use_components = tuple(use_components) if use_components is not None else None
        self.verbose = bool(verbose)
        self._log_fn = log_fn
        self._nlp_override = nlp
        self.project_name = project_name

    # ------------------------- internal logging -------------------------

    def _log(self, msg: str) -> None:
        if self._log_fn:
            try:
                self._log_fn(msg)
                return
            except Exception:
                pass
        if self.verbose:
            print(msg, flush=True)

    # ------------------------- spaCy loading ----------------------------

    @cache
    def get_nlp(self) -> "spacy.language.Language":
        if self._nlp_override is not None:
            return self._nlp_override

        requested = ", ".join(self.use_components) if self.use_components else "all"
        self._log(f"{self.project_name} Loading spaCy model '{self.model_id}' for {self.language} (using: {requested})...")

        def _prune_components(nlp_obj: "spacy.language.Language") -> "spacy.language.Language":
            if self.use_components is None:
                return nlp_obj
            keep = set(self.use_components)
            to_remove = [p for p in nlp_obj.pipe_names if p not in keep]
            for name in to_remove:
                nlp_obj.remove_pipe(name)
            self._log(
                f"{self.project_name} Active components: {', '.join(nlp_obj.pipe_names) or 'none'} "
                f"(removed: {', '.join(to_remove) or 'none'})."
            )
            return nlp_obj

        try:
            nlp = spacy.load(self.model_id)
            nlp = _prune_components(nlp)
            self._log(f"{self.project_name} Loaded '{self.model_id}'.")
            return nlp
        except OSError:
            self._log(f"{self.project_name} Model '{self.model_id}' not found. Attempting to download...")
            try:
                from spacy.cli import download as spacy_download
                self._log(f"{self.project_name} Downloading '{self.model_id}'.")
                spacy_download(self.model_id)
                self._log(f"{self.project_name} Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to install spaCy model '{self.model_id}': {e}")

            try:
                nlp = spacy.load(self.model_id)
                nlp = _prune_components(nlp)
                self._log(f"{self.project_name} Successfully installed and loaded '{self.model_id}'.")
                return nlp
            except Exception as e:
                raise RuntimeError(f"Installed spaCy model '{self.model_id}', but failed loading it: {e}")

    # ------------------------- Core utilities ---------------------------

    def tokenize(self, text: str):
        return list(self.get_nlp()(text))

    def sents(self, text: str) -> List[str]:
        doc = self.get_nlp()(text)
        return [s.text for s in doc.sents] if doc.has_annotation("SENT_START") else [doc.text]

    def pos_tags(self, text: str) -> List[Tuple[str, str]]:
        doc = self.get_nlp()(text)
        return [(t.text, t.pos_) for t in doc]

    def lemmas(self, text: str) -> List[Tuple[str, str]]:
        doc = self.get_nlp()(text)
        return [(t.text, t.lemma_) for t in doc]

    def doc_as_dicts(self, text: str) -> List[Dict[str, Any]]:
        doc = self.get_nlp()(text)
        return [
            {
                "text": t.text,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "tag": t.tag_,
                "dep": t.dep_,
                "idx": t.idx,
                "is_alpha": t.is_alpha,
                "is_punct": t.is_punct,
                "is_stop": t.is_stop,
            }
            for t in doc
        ]

    def pipe(self, texts: Iterable[str], batch_size: int = 1000, n_process: int = 1, as_tuples: bool = False):
        return self.get_nlp().pipe(texts, batch_size=batch_size, n_process=n_process, as_tuples=as_tuples)
