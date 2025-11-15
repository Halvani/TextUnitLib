from __future__ import annotations

from enum import Enum, auto
import itertools
from pathlib import Path
import re
from typing import Any, Callable, Iterable, List, Optional, Union, Set, FrozenSet, Tuple
from functools import lru_cache
from collections import Counter
import warnings

import regex
import spacy
import emoji
import datefinder
from urlextract import URLExtract
from email_scraper import scrape_emails

from . import lib_functions
from .nlp_handler import NlpHandler, SpacyModelSize
import nltk

class TextUnit:
    # Define Enum aliases...
    SpacyModelSize = SpacyModelSize
    ResultingGrams = lib_functions.ResultingGrams    
    
    # Enums used for method arguments
    class Language(Enum):
        English = auto()
        German = auto()
        UnsupportedLanguage = auto()

    class Tokenization(Enum):
        Whitespace = auto()
        WhitespacePunctuation = auto()
        SpacyTokens = auto()
        SpacyStringTokens = auto()

    class AlphaTokens(Enum):
        # Note, for any of the following strategies, *inner* hyphens are preserved as they represent valid parts of words.
        StripSurroundingPuncts = auto()  # Remove punctuation marks on the left and right of the token
        Greedy = auto()  # Scans each character within a token and keeps it in case it is a letter
        OnlyAlphaLetters = auto()  # Check the token as a whole if it consists solely of letters

    class NumeralType(Enum):
        Integers = auto()
        Floats = auto()
        Decimals_0_to_9 = auto()  # Considers only tokens consisting of digits in [0;9]
        Decimals = auto()
        Digits = auto()
        Numerals = auto()
        SpellingOutNumbers = auto()

    class PostagType(Enum):
        Universal = auto()
        Finegrained = auto()

    class WordHeuristic(Enum):
        Postags = auto()
        AlphaTokens = auto()
        NonGibberish = auto()

    class FunctionWordCategoryEn(Enum):
        Conjunctions = auto()
        AuxiliaryVerbs = auto()
        Determiners = auto()
        Prepositions = auto()
        Pronouns = auto()
        Quantifiers = auto()

    def __init__(
        self,
        language: Optional[Language] = None,
        *,
        # You may pass exactly ONE of the following three:
        nlp: Optional[spacy.language.Language] = None,
        nlp_handler: Optional["NlpHandler"] = None,     # type: ignore[name-defined]
        model_id: Optional[Union[str, "SpacyModelSize"]] = None,  # type: ignore[name-defined]
        resources_basepath: Union[str, Path] = "textunitlib/LinguisticResources",
        log_fn=None):
        """
        Create a TextUnit tied to a spaCy pipeline.

        Resolution order for spaCy pipeline:
          1) If `nlp` (spacy.Language) is provided, use it as-is.
          2) Else if `nlp_handler` is provided, use `nlp_handler.get_nlp()`.
          3) Else create a new NlpHandler with the given language (default English)
             and optional model_id and use its pipeline.

        Args:
            language: TextUnit.Language (default: English if None).
            nlp: Preconstructed spaCy pipeline (wins over everything else).
            nlp_handler: An existing NlpHandler instance to source the pipeline.
            model_id: If constructing an NlpHandler, you can force a specific spaCy model
                      (string or SpacyModelSize).
            resources_basepath: Root folder for linguistic resources.
            log_fn: Optional logger callable for diagnostics.
        """
        self._log = (lambda m: log_fn(m)) if callable(log_fn) else (lambda m: None)

        # Default language: English
        self.__language: TextUnit.Language = self.Language.English if language is None else language

        # Validate language early
        if self.__language not in {self.Language.English, self.Language.German}:
            raise ValueError(f"Unsupported language '{self.__language.name}'. Supported: English, German.")

        # Resolve spaCy pipeline
        if nlp is not None:
            if not isinstance(nlp, spacy.Language):
                raise TypeError("`nlp` must be a spaCy Language instance.")
            self.__nlp = nlp
            self._log("[TextUnit] Using caller-provided spaCy pipeline.")
            self.__original_nlp_ruleset = getattr(nlp.tokenizer, "rules", {})
            self.__nlp_handler = None
        else:
            # Try provided NlpHandler first
            if nlp_handler is not None:
                # Basic duck-typing to avoid hard dependency if import path differs
                if not hasattr(nlp_handler, "get_nlp"):
                    raise TypeError("`nlp_handler` must provide a .get_nlp() method.")
                self.__nlp_handler = nlp_handler
                self.__nlp = nlp_handler.get_nlp()
                self.__original_nlp_ruleset = getattr(self.__nlp.tokenizer, "rules", {})
                self._log("[TextUnit] Using spaCy pipeline from provided NlpHandler.")
            else:
                # Construct a new NlpHandler with matching language
                # Import here to avoid circular import at module load time             
                from .nlp_handler import NlpHandler as _NlpHandler, SpacyModelSize as _SpacyModelSize

                lang_hint = self._to_nlp_handler_language(self.__language)
                handler_kwargs = {"language": lang_hint}                
                
                if model_id is not None:
                    handler_kwargs["model_id"] = model_id
                self.__nlp_handler = _NlpHandler(**handler_kwargs)
                self.__nlp = self.__nlp_handler.get_nlp()
                self.__original_nlp_ruleset = getattr(self.__nlp.tokenizer, "rules", {})
                self._log("[TextUnit] Constructed NlpHandler and loaded spaCy pipeline.")

        # Resources — language-dependent
        base = Path(resources_basepath)
        if self.__language == self.Language.English:
            self.__vowels = self.__load_resource(base / "en" / "vowels" / "vowels.txt")
            self.__contractions = self.__load_resource(base / "en" / "contractions" / "contractions.txt")
            self.__stopwords = self.__load_resource(base / "en" / "stopwords" / "stopwords.txt")

            fw_base_en = base / "en" / "functionwords"
            # All English function words
            self.__functionwords = self.__load_resource(fw_base_en / "all.txt")

            # English function word categories
            self.en_functionwords_auxiliary_verbs = self.__load_resource(fw_base_en / "auxiliary_verbs.txt")
            self.en_functionwords_conjunctions = self.__load_resource(fw_base_en / "conjunctions.txt")
            self.en_functionwords_determiners = self.__load_resource(fw_base_en / "determiners.txt")
            self.en_functionwords_prepositions = self.__load_resource(fw_base_en / "prepositions.txt")
            self.en_functionwords_pronouns = self.__load_resource(fw_base_en / "pronouns.txt")
            self.en_functionwords_quantifiers = self.__load_resource(fw_base_en / "quantifiers.txt")

        elif self.__language == self.Language.German:
            self.__vowels = self.__load_resource(base / "de" / "vowels" / "vowels.txt")
            self.__contractions = self.__load_resource(base / "de" / "contractions" / "contractions.txt")
            self.__stopwords = self.__load_resource(base / "de" / "stopwords" / "stopwords.txt")

            fw_all = base / "de" / "functionwords" / "all.txt"
            # Once we have the German function words, we load them here
            self.__functionwords = self.__load_resource(fw_all) if fw_all.exists() else frozenset()

        # Language-independent resources
        self.__punctuation = self.__load_resource(base / "independent" / "punctuation.txt")

    # ------------------------- helpers -------------------------

    @staticmethod
    def _to_nlp_handler_language(lang: "TextUnit.Language") -> Union[str, Enum]:
        """
        Convert TextUnit.Language to the language hint expected by NlpHandler.
        By default, the NlpHandler normalizes 'english'/'german' strings, so we return those.
        """
        if lang == TextUnit.Language.English:
            return "english"
        if lang == TextUnit.Language.German:
            return "german"
        # Should not be reached due to validation in __init__
        raise ValueError(f"Unsupported language: {lang}")

    @staticmethod
    def __load_resource(path: Union[str, Path]) -> frozenset[str]:
        """
        Load a simple newline-delimited resource file into a frozenset of strings.
        Empty lines are ignored; lines are stripped.
        """
        p = Path(path)
        if not p.exists():
            return frozenset()
        lines = p.read_text(encoding="utf8").splitlines()
        items = [ln.strip() for ln in lines if ln.strip()]
        return frozenset(items)
    
   
    @lru_cache(maxsize=None)
    def __load_resource(self, filepath_resource: Union[str, Path], load_as_frozenset: bool = True) -> Union[List[str], FrozenSet[str]]:
        """
        Load and cache a linguistic resource from a file.

        This method reads a text file containing newline-separated entries (e.g., stopwords)
        and returns its content either as a list or a frozenset. The result is cached for
        repeated calls with the same arguments.

        Args:
            filepath_resource (Union[str, Path]): Path to the resource file.
            load_as_frozenset (bool): If True, return a frozenset (default). Otherwise, return a list.

        Returns:
            Union[List[str], FrozenSet[str]]: Cleaned, stripped, non-empty lines from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If filepath_resource is None.

        Example:
            >>> __load_resource(Path("LinguisticResources/independent/apostrophes.txt"))
            frozenset({"’", "'"})
        """
        if filepath_resource is None:
            raise ValueError("`filepath_resource` must not be None.")

        path = Path(filepath_resource)

        if not path.exists():
            raise FileNotFoundError(f"Resource file not found: {path}")

        # Read, strip, and ignore empty lines
        lines = [ln.strip() for ln in path.read_text(encoding="utf8").splitlines() if ln.strip()]

        return frozenset(lines) if load_as_frozenset else lines


    def __expand_with_duplicates(self, scraped_strings, text):
        """
        Expand a set or list of strings according to their occurrences in a given text.

        Each string in scraped_strings is counted in the text. The function returns a list
        containing every string as many times as it appears in the text. Strings that do not
        occur in the text are ignored.

        Parameters
        ----------
        scraped_strings : iterable of str
            A collection of strings that should be searched in the text.
        text : str
            The text in which occurrences of the strings are counted.

        Returns
        -------
        list of str
            A list containing each string repeated according to how often it appears in the text.
        """
        result = []
        for s in scraped_strings:
            count = text.count(s)
            if count > 0:
                result.extend([s] * count)
        return result   
    # ------------------------- public properties -------------------------

    @property
    def nlp(self) -> spacy.language.Language:
        """Access the underlying spaCy pipeline."""
        return self.__nlp

    @property
    def language(self) -> "TextUnit.Language":
        """The selected language."""
        return self.__language

    @property
    def prop_vowels(self) -> frozenset[str]:
        return self.__vowels

    @property
    def prop_contractions(self) -> frozenset[str]:
        return self.__contractions

    @property
    def prop_stopwords(self) -> frozenset[str]:
        return self.__stopwords

    @property
    def prop_functionwords(self) -> frozenset[str]:
        return self.__functionwords

    @property
    def prop_punctuation(self) -> frozenset[str]:
        return self.__punctuation
    
    
    def __translate_language_to_code(self, language: Language) -> str:
        """
        Return the ISO-style language code ('en', 'de') for the given Language enum.
        Raises ValueError if the language is unsupported.
        """
        lang_map = {
            self.Language.English: "en",
            self.Language.German: "de",
        }

        code = lang_map.get(language)
        if code is None:
            raise ValueError(f"Unsupported language: {language.name}")
        return code
    
    
    def __none_or_empty(self, data: Any) -> bool:
        """Return True if data is None or empty (supports str, list, tuple, set)."""
        if data is None:
            return True
        if isinstance(data, str):
            data = data.strip()
        if isinstance(data, (str, list, tuple, set)):
            return len(data) == 0
        raise ValueError(f"Unsupported type: {type(data).__name__}")
        

    def __strip_surrounding_puncts(self, token: str, apostrophes: Optional[Iterable[str]] = None) -> str:
        """
        Strip surrounding punctuation while preserving inner hyphens and apostrophes.
        The right boundary may end with an apostrophe (e.g., "jones'").
        If an illegal inner character (not letter, hyphen, or apostrophe) is found,
        return an empty string.
        """
        if not isinstance(token, str):
            raise TypeError(f"`token` must be str, got {type(token).__name__}")

        # Early outs
        s = token.strip()
        if not s:
            return ""

        apos = set(apostrophes) if apostrophes else {"’", "'"}

        n = len(s)

        # Find first letter from the left
        i = 0
        while i < n and not s[i].isalpha():
            i += 1
        if i >= n:
            return ""  # no letters at all

        # Find last allowed char from the right (letter or apostrophe)
        j = n - 1
        while j >= i and not (s[j].isalpha() or s[j] in apos):
            j -= 1
        if j < i:
            return ""

        candidate = s[i:j + 1]

        # Validate inner chars: letters, hyphen, or apostrophes only
        for ch in candidate:
            if ch.isalpha() or ch == "-" or ch in apos:
                continue
            return ""

        return candidate
    
    
    # Core functions===========================================================================


    def characters(self, text: str, drop_whitespaces: bool = False) -> List[str]:
        """
        Return a list of characters from the input text, optionally excluding whitespace.

        Args:
            text (str): The input text.
            drop_whitespaces (bool, optional): If True, all whitespace characters are excluded. Defaults to False.

        Returns:
            List[str]: List of extracted characters.

        Examples:
            >>> tu = TextUnit()
            >>> tu.characters(" Text Unit Lib ")
            [' ', 'T', 'e', 'x', 't', ' ', 'U', 'n', 'i', 't', ' ', 'L', 'i', 'b', ' ']

            >>> tu.characters(" Text Unit Lib ", drop_whitespaces=True)
            ['T', 'e', 'x', 't', 'U', 'n', 'i', 't', 'L', 'i', 'b']
        """
        return [c for c in text if not (drop_whitespaces and c.isspace())]


    def spaces(self, text: str) -> List[str]:
        """
        Return a list of whitespace characters found in the given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            List[str]: Whitespace characters extracted from the text.

        Example:
            >>> tu = TextUnit()
            >>> tu.spaces(" Text\tUnit  \tLib\t ")
            [' ', '\t', ' ', ' ', '\t', '\t', ' ']

        Note:
            This method uses the `characters` method with `drop_whitespaces=False`
            and filters the result for whitespace characters.
        """
   
        return [c for c in self.characters(text, drop_whitespaces=False) if c.isspace()]

    def punctuation_marks(self, text: str) -> List[str]:
        """
        Return a list of punctuation marks found in the given text.

        Args:
            text (str): The input text.

        Returns:
            List[str]: Punctuation marks extracted from the text.

        Example:
            >>> tu = TextUnit()
            >>> tu.punctuation_marks("Hello, TextUnitLib! How are you?")
            [',', '!', '?']

        Note:
            This method uses the `characters` method to extract characters,
            then filters those found in the instance attribute `__punctuation`.
        """
        
        return [c for c in self.characters(text) if c in self.__punctuation]


    def vowels(self, text: str) -> List[str]:
        """
        Return a list of vowels found in the given text.

        Vowels are defined by the instance attribute `__vowels` and are
        matched case-insensitively.

        Args:
            text (str): The input text.

        Returns:
            List[str]: All vowels occurring in the text.

        Example:
            >>> tu = TextUnit()
            >>> tu.vowels("Hello, how are you?")
            ['e', 'o', 'o', 'a', 'e', 'y', 'o', 'u']
        """
        return [c for c in self.characters(text) if c.lower() in self.__vowels]


    def letters(self, text: str) -> List[str]:
        """
        Return a list of alphabetic characters from the input text.

        Args:
            text (str): The input text.

        Returns:
            List[str]: Alphabetic characters from the text.

        Example:
            >>> tu = TextUnit()
            >>> tu.letters("TextUnitLib --> Released @2024!")
            ['T', 'e', 'x', 't', 'U', 'n', 'i', 't', 'L', 'i', 'b', 'R', 'e', 'l', 'e', 'a', 's', 'e', 'd']
        """
        return [c for c in self.characters(text) if c.isalpha()]


    def digits(self, text: str) -> List[str]:
        """
        Return a list of digit characters from the input text.

        Args:
            text (str): The input text.

        Returns:
            List[str]: Digits found in the text.

        Example:
            >>> tu = TextUnit()
            >>> tu.digits("abc123xyz456")
            ['1', '2', '3', '4', '5', '6']
        """
        return [c for c in self.characters(text) if c.isdigit()]


    def textunit_ngrams(self, text_units: Union[str, List[str]], n: int,
        sep: str = " ", strip_spaces: bool = False, resulting_grams: ResultingGrams = ResultingGrams.CHARACTERS) -> List[str]:
        """
        Thin wrapper around the library function in lib_functions.textunit_ngrams.
        """
        return lib_functions.textunit_ngrams(
            text_units=text_units,
            n=n,
            sep=sep,
            strip_spaces=strip_spaces,
            resulting_grams=resulting_grams)
        
        
    def char_ngrams(self, text: str, n: int, strip_spaces: bool = False) -> List[str]:
        """
        Generate character n-grams from the input text.

        Args:
            text (str): The input text.
            n (int): The n-gram length.
            strip_spaces (bool, optional): If True, leading and trailing spaces
                are removed from each n-gram. Defaults to False.

        Returns:
            List[str]: A list of character n-grams.

        Example:
            >>> tu = TextUnit()
            >>> tu.char_ngrams("Hello World!", 3)
            ['Hel', 'ell', 'llo', 'lo ', 'o W', ' Wo', 'Wor', 'orl', 'rld', 'ld!']

            >>> tu.char_ngrams("Hello World!", 3, strip_spaces=True)
            ['Hel', 'ell', 'llo', 'lo', 'o W', 'Wo', 'Wor', 'orl', 'rld', 'ld!']
        """
        return self.textunit_ngrams(
            text_units=text,
            n=n,
            sep="",  # no separator between characters
            strip_spaces=strip_spaces,
            resulting_grams=self.ResultingGrams.CHARACTERS,
        )
        

    def char_ngrams_range(self, text: str, n_from: int = 3, n_to: int = 5) -> List[str]:
        """
        Generate character n-grams within a specified range from the given text.

        Args:
            text (str): The input text from which n-grams will be generated.
            n_from (int, optional): The minimum n-gram length. Defaults to 3.
            n_to (int, optional): The maximum n-gram length. Defaults to 5.

        Returns:
            List[str]: All generated character n-grams within the specified range.

        Raises:
            ValueError: If n_from is less than 1 or n_to is smaller than n_from.

        Example:
            >>> tu = TextUnit()
            >>> tu.char_ngrams_range("example text")
            ['exa', 'xam', 'amp', 'mpl', 'ple', 'le ', 'e t', ' te', 'tex', 'ext',
             'exam', 'xamp', 'ampl', 'mple', 'ple ', 'le t', 'e te', ' tex', 'text',
             'examp', 'xampl', 'ample', 'mple ', 'ple t', 'le te', 'e tex', ' text']
        """
        if n_from < 1:
            raise ValueError("n_from must be at least 1.")
        if n_to < n_from:
            raise ValueError("n_to must be greater than or equal to n_from.")

        range_ngrams: List[str] = []
        for n in range(n_from, n_to + 1):
            range_ngrams.extend(self.char_ngrams(text=text, n=n))

        return range_ngrams


    def emojis(self, text: str, decompose_emoji_clusters: bool = False, demojize: bool = False) -> List[str]:
        """
        Extracts emojis from the given text and provides options for emoji manipulation.

        Args:
            text (str): The input text from which emojis are to be extracted.
            decompose_emoji_clusters (bool, optional): If True, decomposes emoji clusters into individual
            (e.g., '👨‍👩‍👦‍👦' --> '👨', '👩', '👦', '👦') emojis. Defaults to False.
            demojize (bool, optional): If True, replaces extracted emojis with their corresponding emoji shortcodes.
                Defaults to False.

        Returns:
            List[str]: A list containing extracted emojis or their corresponding shortcodes (based on given options).

        Note:
            If `decompose_emoji_clusters` is set to True, the function decomposes emoji clusters into individual emojis.
            If `demojize` is set to True, the function replaces extracted emojis with their corresponding shortcodes.

        Example:
            >>> emojis("Time for a little break with healthy and delicious snacks 🍌🥭🥝🍊🍎")
            ['🍌', '🥭', '🥝', '🍊', '🍎']

            >>> emojis("Fine👍 I'm currently visiting my 👩‍👧‍👧!", decompose_emoji_clusters= True)
            ['👍', '👩', '👧', '👧']

            >>> emojis("Hello TextUnitLib! 😊🌍🚀", demojize=True)
            ['smiling_face_with_smiling_eyes', 'globe_showing_Europe-Africa', 'rocket']
        """

        lang_code = self.__translate_language_to_code(self.__language)

        if decompose_emoji_clusters:
            emojis = [c for c in text if emoji.is_emoji(c)]
            return [emoji.demojize(c, language=lang_code) for c in emojis] if demojize else emojis

        graphemes = regex.findall(r"\X", text)
        emojis = [grapheme for grapheme in graphemes if emoji.is_emoji(grapheme)]
        return [emoji.demojize(c, language=lang_code) for c in emojis] if demojize else emojis
        

    def tokens(
        self,
        text: str,
        strategy: Tokenization = Tokenization.Whitespace,
        sep: Optional[str] = None) -> List[Union[str, spacy.tokens.token.Token]]:
        """
        Tokenize text using the specified strategy.

        Args:
            text (str): The input text to tokenize.
            strategy (Tokenization, optional): Tokenization strategy.
                - Whitespace: split on whitespace or a custom separator
                - WhitespacePunctuation: split on words and punctuation
                - SpacyTokens: return spaCy Token objects
                - SpacyStringTokens: return token texts from spaCy
            sep (str, optional): Custom separator used only with the Whitespace strategy.
                If None, split on arbitrary whitespace.

        Returns:
            List[Union[str, Token]]: A list of tokens according to the chosen strategy.

        Raises:
            RuntimeError: If a spaCy based strategy is used but the pipeline is not initialized.
            ValueError: If an unknown tokenization strategy is provided.
        """
        if strategy is self.Tokenization.Whitespace:
            # Default split uses arbitrary whitespace when sep is None
            return text.split() if sep is None else text.split(sep)

        if strategy is self.Tokenization.WhitespacePunctuation:
            return nltk.wordpunct_tokenize(text)

        # spaCy based strategies require an initialized pipeline
        if self.__nlp is None and strategy in {self.Tokenization.SpacyTokens, self.Tokenization.SpacyStringTokens}:
            raise RuntimeError("spaCy pipeline is not initialized but a spaCy tokenization strategy was requested.")

        if strategy is self.Tokenization.SpacyTokens:
            return list(self.__nlp(text))

        if strategy is self.Tokenization.SpacyStringTokens:
            return [t.text for t in self.__nlp(text)]

        raise ValueError(f"Unsupported tokenization strategy: {strategy}")


    def alphabetic_tokens(
        self,
        text: str,
        lowered: bool = False,
        strategy: AlphaTokens = AlphaTokens.StripSurroundingPuncts,
        min_length: int = 1) -> List[str]:
        """
        Extract tokens consisting of letters with optional inner hyphens or apostrophes.
        Inner hyphens or apostrophes are allowed only when not at the beginning or end.

        Parameters:
            text: Input text.
            lowered: Lowercase the extracted tokens.
            strategy: One of AlphaTokens strategies.
            min_length: Minimum accepted token length.

        Returns:
            List[str]: Tokens that match the rules for the chosen strategy.
        """
        # Pattern: one or more Unicode letters, optionally repeated groups of
        # (hyphen or apostrophe + one or more letters). This prevents leading/trailing
        # hyphens/apostrophes while allowing inner ones.
        pattern_full = regex.compile(r"\p{L}+(?:[’'-]\p{L}+)*", flags=regex.VERSION1)

        out: List[str] = []

        # Helper to normalize tokens returned by self.tokens (they may be spaCy Token objects)
        def token_text(t) -> str:
            return t.text if hasattr(t, "text") else str(t)

        if strategy is self.AlphaTokens.OnlyAlphaLetters:
            # Only accept entire tokens that match the pattern (token boundaries preserved)
            for tok in self.tokens(text):
                s = token_text(tok)
                if pattern_full.fullmatch(s):
                    cand = s.lower() if lowered else s
                    if len(cand) >= min_length:
                        out.append(cand)
            return out

        # For StripSurroundingPuncts and Greedy, scan the text for valid letter sequences.
        # This approach is robust across punctuation and whitespace.
        candidates = pattern_full.findall(text)

        if strategy is self.AlphaTokens.StripSurroundingPuncts:
            # pattern already strips surrounding punctuation by design; keep as-is
            keep = candidates
        else:  # Greedy
            # Greedy: keep letter/inner-char runs anywhere; pattern already matches inner groups,
            # but to emulate greedy behaviour we also accept runs of letters+inner chars.
            keep = candidates

        for cand in keep:
            if len(cand) >= min_length:
                out.append(cand.lower() if lowered else cand)

        return out


    def vocabulary(self, text: str, strategy: AlphaTokens = AlphaTokens.StripSurroundingPuncts) -> List[str]:
        """
        Extract a sorted, unique vocabulary of alphabetic tokens from the input text.

        This method tokenizes the text using the specified strategy to extract alphabetic tokens
        (words), converts them to lowercase, removes duplicates, and returns them in sorted order.

        Args:
            text (str): The input text from which to extract vocabulary.
            strategy (AlphaTokens, optional): The token extraction strategy to use.
                - StripSurroundingPuncts: Remove surrounding punctuation (default)
                - Greedy: Keep character runs that contain letters
                - OnlyAlphaLetters: Only full tokens consisting entirely of letters
                Defaults to AlphaTokens.StripSurroundingPuncts.

        Returns:
            List[str]: A sorted list of unique lowercase alphabetic tokens (vocabulary items).

        Example:
            >>> tu = TextUnit()
            >>> tu.vocabulary("Hello, world! Hello TextUnit.")
            ['hello', 'textunit', 'world']
        """
        alpha_tokens = self.alphabetic_tokens(text, lowered=True, strategy=strategy)
        return sorted(set(alpha_tokens))


    def regex_strings(self, text: str, pattern: str, flags: int = 0) -> List[str]:
        """
        Extract all non-overlapping substrings from text matching the given regex pattern.ern.
    
        This method uses the `regex` module (not the standard `re`) to find all matchesches
        of the provided pattern in the input text. The regex module provides bettertter
        Unicode support and additional features compared to the standard library.ary.
    
        Args:rgs:
            text (str): The input text to search within.hin.
            pattern (str): The regular expression pattern to match.tch.
                Must not be None or empty. See https://pypi.org/project/regex/ for pattern syntax.tax.
            flags (int, optional): Regex compilation flags to modify matching behavior.ior.
                Common flags: regex.IGNORECASE, regex.MULTILINE, regex.DOTALL, regex.VERBOSE.OSE.
                See https://docs.python.org/3/library/re.html#flags for details.ils.
                Defaults to 0 (no flags).gs).
    
        Returns:rns:
            List[str]: A list of all non-overlapping pattern matches found in the text.ext.
                Returns an empty list if no matches are found or if the pattern is empty.pty.
    
        Raises:ses:
            regex.error: If the pattern is invalid.

        Example:            Example:
            >>> tu = TextUnit()>>> tu.regex_strings("abc123def456ghi", r"\\d+")rings("abc123def456ghi", r"\\d+")
            ['123', '456']

            >>> tu.regex_strings("Hello WORLD hello", r"\\w+", flags=regex.IGNORECASE)      
            >>> tu.regex_strings("Hello WORLD hello", r"\\w+", flags=regex.IGNORECASE)
            ['Hello', 'WORLD', 'hello']
        """
        if self.__none_or_empty(pattern):
            return []
        return regex.findall(pattern, text, flags=flags)


    def legomenon_units(self, text_units: List[str], n: int) -> List[str]:
        """
        Return text units that occur exactly `n` times in the input list.

        Args:
            text_units (List[str]): The list of text units (e.g., words, POS tags, or phrases) to analyze.
            n (int): The target frequency of occurrences.

        Returns:
            List[str]: Text units that appear exactly `n` times.

        Example:
            >>> tu = TextUnit()
            >>> text_units = ["apple", "banana", "apple", "orange", "banana", "apple"]
            >>> tu.legomenon_units(text_units, 2)
            ['banana']
        """
        return [unit for unit, freq in Counter(text_units).items() if freq == n]


    def hapax_legomenon_units(self, text_units: List[str]) -> List[str]:
        """Return text units that appear only once (hapax legomena)."""
        return self.legomenon_units(text_units, n=1)


    def dis_legomenon_units(self, text_units: List[str]) -> List[str]:
        """Return text units that appear exactly twice (dis legomena)."""
        return self.legomenon_units(text_units, n=2)


    def tris_legomenon_units(self, text_units: List[str]) -> List[str]:
        """Return text units that appear exactly three times (tris legomena)."""
        return self.legomenon_units(text_units, n=3)


    def dates(self, text: str, output_format: str = "%d.%m.%Y", preserve_input_format: bool = False) -> List[str]:
        """
        Extract dates from the given text and return them either in a unified format
        or in their original textual form.

        Parameters
        ----------
        text : str
            The input text containing dates.
        output_format : str, optional
            The desired output format for the dates when preserve_input_format is False.
            Default is "%d.%m.%Y".
        preserve_input_format : bool, optional
            If True the function returns date expressions exactly as they are matched
            by the underlying date extraction library. These matches may include
            surrounding words and may also include strings that are not strict
            date tokens. This option preserves the original substring but it is not
            guaranteed to produce clean or unambiguous date formats.

        Returns
        -------
        List[str]
            A list of extracted dates. The dates are either formatted uniformly or
            preserved in their original matched form depending on the chosen settings.
        """
        if preserve_input_format:
            return [original for _, original in datefinder.find_dates(text, source=True)]

        matches = datefinder.find_dates(text)
        return [d.strftime(output_format) for d in matches]

 
    def urls(self, text: str, extract_with_duplicates: bool = True) -> List[str]:
        """
        Extract URLs from the given text.

        Parameters
        ----------
        text : str
            The input text from which URLs are extracted.
        extract_with_duplicates : bool, optional
            If True the result contains each URL as many times
            as it appears in the text. If False the result
            contains each URL only once.

        Returns
        -------
        List[str]
            A list of URLs extracted from the text. The list
            preserves duplicates if extract_with_duplicates is True
            otherwise the result contains unique URLs only.

        Example
        -------
        >>> urls("Go to https://a.com and also https://a.com")
        ['https://a.com', 'https://a.com']

        >>> urls("Go to https://a.com and also https://a.com", extract_with_duplicates=False)
        ['https://a.com']
        """
        extractor = URLExtract()
        urls = list(extractor.gen_urls(text))

        if extract_with_duplicates:
            return urls
        return list(set(urls))


    def email_addresses(self, text: str, extract_with_duplicates: bool = True) -> List[str]:
        """
        Extract email addresses from the given text.

        Parameters
        ----------
        text : str
            The input text from which email addresses are extracted.
        extract_with_duplicates : bool, optional
            If True the result contains each email address as many times
            as it appears in the text. If False the result contains each
            email address only once.

        Returns
        -------
        List[str]
            A list of email addresses found in the text. The list contains
            duplicates if extract_with_duplicates is True otherwise it
            contains unique addresses only.

        Examples
        --------
        >>> email_addresses("Contact us at john.doe@example.com or support@example.com")
        ['john.doe@example.com', 'support@example.com']

        >>> email_addresses("Mail x@example.com twice x@example.com", extract_with_duplicates=True)
        ['x@example.com', 'x@example.com']

        >>> email_addresses("Mail x@example.com twice x@example.com", extract_with_duplicates=False)
        ['x@example.com']
        """
        scraped_mails = scrape_emails(text)
        unique_emails = list(scraped_mails)

        if extract_with_duplicates:
            return self.__expand_with_duplicates(unique_emails, text)
        return unique_emails



    def sentences(self, text: str, remove_empty_lines: bool = True) -> List[str]:
        """
        Segment the input text into sentences using the spaCy pipeline.

        Args:
            text (str): The input text to segment.
            remove_empty_lines (bool, optional): If True, exclude sentences that are empty
                or contain only whitespace. Defaults to True.

        Returns:
            List[str]: A list of sentence strings extracted from the text.

        Raises:
            RuntimeError: If the spaCy pipeline is not initialized.

        Example:
            >>> tu = TextUnit()
            >>> tu.sentences("Hello world. This is TextUnit.")
            ['Hello world.', 'This is TextUnit.']
        """
        if self.__nlp is None:
            raise RuntimeError("spaCy pipeline is not initialized.")

        doc = self.__nlp(text)

        sentences = [s.text.strip() for s in doc.sents]
        if remove_empty_lines:
            sentences = [s for s in sentences if s]
        return sentences

    def postags(self,
                text: str,
                postag_type: PostagType = PostagType.Universal,
                combine_with_token: bool = False,
                combine_sep: Optional[str] = None,
                tags_to_consider: Optional[List[str]] = None) -> List[Union[str, tuple]]:
        """
        Extract part-of-speech (POS) tags from the input text using the spaCy pipeline.

        Args:
            text (str): The input text to analyze.
            postag_type (PostagType, optional): The POS tag scheme to use.
                - Universal: Universal POS tags (default, coarse-grained)
                - Finegrained: Language-specific fine-grained POS tags
                Defaults to PostagType.Universal.
            combine_with_token (bool, optional): If True, combine each token with its POS tag.
                If False, return only POS tags. Defaults to False.
            combine_sep (str, optional): Separator used when combining tokens and tags.
                Only used if combine_with_token=True.
                - If None: returns (token, tag) tuples
                - If provided: returns "token<sep>tag" strings
                Defaults to None.
            tags_to_consider (List[str], optional): If provided, only extract tokens/tags
                that match one of the specified POS tags. If None, include all tags.
                Defaults to None.

        Returns:
            List[Union[str, tuple]]: POS tags or combinations depending on parameters:
                - combine_with_token=False: List of POS tag strings
                - combine_with_token=True, combine_sep=None: List of (token, tag) tuples
                - combine_with_token=True, combine_sep=str: List of "token<sep>tag" strings

        Raises:
            RuntimeError: If the spaCy pipeline is not initialized.
            ValueError: If postag_type is not a valid PostagType enum member.

        Example:
            >>> tu = TextUnit()
            >>> tu.postags("Hello world!")
            ['INTJ', 'NOUN', 'PUNCT']

            >>> tu.postags("Hello world!", combine_with_token=True, combine_sep="_")
            ['Hello_INTJ', 'world_NOUN', '!_PUNCT']

            >>> tu.postags("Hello world!", combine_with_token=True)
            [('Hello', 'INTJ'), ('world', 'NOUN'), ('!', 'PUNCT')]

            >>> tu.postags("Hello world!", tags_to_consider=['NOUN', 'VERB'])
            ['NOUN']
        """
        if self.__nlp is None:
            raise RuntimeError("spaCy pipeline is not initialized.")

        if postag_type not in {self.PostagType.Universal, self.PostagType.Finegrained}:
            raise ValueError(f"Invalid postag_type: {postag_type}. Must be Universal or Finegrained.")

        result = []
        spacy_tokens = self.tokens(text, strategy=self.Tokenization.SpacyTokens)

        for t in spacy_tokens:
            token = t.text
            postag = t.pos_ if postag_type == self.PostagType.Universal else t.tag_

            # Skip if filtering by specific tags
            if tags_to_consider is not None and postag not in tags_to_consider:
                continue

            # Build result based on combine_with_token flag
            if combine_with_token:
                # Combine token and tag
                if self.__none_or_empty(combine_sep):
                    # Return as tuple if no separator provided
                    result.append((token, postag))
                else:
                    # Return as combined string with separator
                    result.append(f"{token}{combine_sep}{postag}")
            else:
                # Return only the POS tag
                result.append(postag)
        return result


    def postag_ngrams(
        self,
        text: str,
        n: int,
        postag_type: PostagType = PostagType.Universal,
        combine_with_token: bool = False,
        combine_sep: Optional[str] = None,
        tags_to_consider: Optional[List[str]] = None,
        ngram_sep: str = " ") -> List[str]:
        """
        Build n-grams over POS tags (or token+POS units) extracted from text.

        Parameters:
            text (str): Input text.
            n (int): n for n-grams (must be >= 1).
            postag_type (PostagType): Use Universal (coarse) or Finegrained (language-specific) tags.
            combine_with_token (bool): If True, use token+tag units; otherwise use tags only.
            combine_sep (Optional[str]): Separator to combine token and tag when combine_with_token=True.
                If None, a default "/" separator is used when converting tuple units to strings.
            tags_to_consider (Optional[List[str]]): If provided, only units with tags in this list are included.
            ngram_sep (str): Separator inserted between units when forming the resulting n-gram strings.

        Returns:
            List[str]: List of n-gram strings composed from POS-based units.

        Raises:
            ValueError: If n < 1 or postag_type is invalid.
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        if postag_type not in {self.PostagType.Universal, self.PostagType.Finegrained}:
            raise ValueError(f"Invalid postag_type: {postag_type}")

        postags = self.postags(
            text=text,
            postag_type=postag_type,
            combine_with_token=combine_with_token,
            combine_sep=combine_sep,
            tags_to_consider=tags_to_consider)

        # Ensure we work with a list of string units
        if combine_with_token:
            # Use the provided separator as-is; only use the default "/" when combine_sep is None.
            # Treating whitespace-only separators as valid is important (e.g. combine_sep=" ").
            sep_used = "/" if combine_sep is None else combine_sep
            units: List[str] = [
                u if isinstance(u, str) else f"{u[0]}{sep_used}{u[1]}" for u in postags
            ]
        else:
            units = [str(u) for u in postags]

        # Build n-grams explicitly to preserve the provided ngram_sep (including spaces)
        if n > len(units):
            return []

        ngrams: List[str] = [ngram_sep.join(units[i : i + n]) for i in range(0, len(units) - n + 1)]
        return ngrams


    def token_ngrams(self, text: str, n: int = 3, sep: str = " ", strip_spaces: bool = False) -> List[str]:
        """
        Generate sliding n-grams over token units extracted from text.

        Tokenizes the input text (spaCy tokens or simple strings), normalizes each
        token to its text form and builds overlapping n-grams (window size `n`,
        step 1). The separator `sep` is used to join tokens inside each n-gram;
        whitespace-only separators (e.g. " ") are accepted. When `strip_spaces`
        is True, tokens are stripped of surrounding whitespace before building n-grams.

        Args:
            text (str): Input text to tokenize.
            n (int): n-gram length (must be >= 1).
            sep (str): Separator used between tokens inside an n-gram (default: single space).
            strip_spaces (bool): If True, strip leading/trailing whitespace from each token.

        Returns:
            List[str]: List of token n-gram strings (empty list if not enough tokens).

        Raises:
            ValueError: If n < 1.

        Examples:
            >>> tu = TextUnit()
            >>> tu.token_ngrams("The quick brown fox", n=2)
            ['The quick', 'quick brown', 'brown fox']

            >>> tu.token_ngrams("The kickoff meeting", n=2, sep="_")
            ['The_kickoff', 'kickoff_meeting']
        """
        if n < 1:
            raise ValueError("n must be >= 1.")

        raw_tokens = self.tokens(text)
        # Normalize tokens to plain strings (handle spaCy Token objects)
        token_texts: List[str] = [t if isinstance(t, str) else getattr(t, "text", str(t)) for t in raw_tokens]

        if strip_spaces:
            token_texts = [tt.strip() for tt in token_texts]

        # remove any empty tokens that may remain after stripping
        token_texts = [tt for tt in token_texts if tt != ""]

        if n > len(token_texts):
            return []

        # explicit sliding-window join to ensure `sep` is always honored
        return [sep.join(token_texts[i : i + n]) for i in range(0, len(token_texts) - n + 1)]


    def textunit_lengths(self,
                         text_units: List[str],
                         lengths_only: bool = False,
                         sort_by_length: bool = False) -> Union[List[int], dict]:
        """
        Analyze lengths of given text units.

        Depending on flags this function either returns the list of lengths for
        each input unit or a mapping from length -> list of units having that length.

        Parameters:
            text_units (List[str]): Sequence of text units (words, tokens, phrases).
            lengths_only (bool): If True return a List[int] with the length of each
                input unit in the original order. If False return a dict mapping
                length -> List[str].
            sort_by_length (bool): When returning the dict, if True the mapping is
                returned with keys sorted in ascending order. The order of units
                inside each list preserves the original input order.

        Returns:
            Union[List[int], dict]: Either a list of lengths (if lengths_only=True)
            or a dict mapping int -> List[str] (if lengths_only=False).

        Raises:
            TypeError: If text_units is not a list/tuple or contains non-string elements.

        Examples:
            >>> tu = TextUnit()
            >>> tu.textunit_lengths(["a", "bb", "ccc"], lengths_only=True)
            [1, 2, 3]
            >>> tu.textunit_lengths(["a", "bb", "ccc"])
            {1: ['a'], 2: ['bb'], 3: ['ccc']}
            >>> tu.textunit_lengths(["aa","b","cc","d"], sort_by_length=True)
            {1: ['b','d'], 2: ['aa','cc']}
        """
        if not isinstance(text_units, (list, tuple)):
            raise TypeError("text_units must be a list or tuple of strings.")
        for u in text_units:
            if not isinstance(u, str):
                raise TypeError("All elements of text_units must be strings.")

        # Fast path: return raw lengths preserving input order
        if lengths_only:
            return [len(u) for u in text_units]

        # Build mapping length -> [units], preserving original order of units
        length_distribution: dict = {}
        for u in text_units:
            length_distribution.setdefault(len(u), []).append(u)

        if sort_by_length:
            return dict(sorted(length_distribution.items(), key=lambda kv: kv[0]))

        return length_distribution


    def stop_words(self, text: str, lowered: bool = False) -> List[str]:
        """
        Extract stop words present in the input text.

        The method extracts alphabetic tokens from `text` and returns those
        tokens that appear in the language-specific stopword list loaded
        for this TextUnit instance.

        Parameters:
            text (str): Input text to inspect.
            lowered (bool): If True, tokens are lowercased before matching and
                the returned tokens are lowercase. If False, matching is performed
                case-insensitively but returned tokens preserve their original case.
                Defaults to False.

        Returns:
            List[str]: Stopword tokens found in `text`, in original order.
                Duplicate occurrences are preserved.

        Raises:
            TypeError: If `text` is not a string.

        Examples:
            >>> tu = TextUnit()
            >>> tu.stop_words("This is a sample sentence with some common stop words.")
            ['is', 'a', 'with', 'some']
            >>> tu.stop_words("This is a SAMPLE sentence.", lowered=True)
            ['is', 'a', 'sample']
        """
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")

        # Extract alphabetic tokens; lowercase them if requested
        alpha_tokens = self.alphabetic_tokens(
            text=text,
            lowered=lowered,
            strategy=self.AlphaTokens.StripSurroundingPuncts,
        )

        if lowered:
            # tokens already lowercased — direct membership test against stopwords set
            return [t for t in alpha_tokens if t in self.__stopwords]

        # case-insensitive match but return tokens in original case
        return [t for t in alpha_tokens if t.lower() in self.__stopwords]


    def top_k_textunits(self, text_units: Iterable[str], topk: int, unique: bool = False) -> List[str]:
        """
        Return the top-k text units by frequency.

        Parameters:
            text_units (Iterable[str]): Iterable of text units (tokens, n-grams, tags, ...).
            topk (int): Number of distinct units to consider (must be >= 1).
            unique (bool): If True, return the distinct units only (one entry per unit).
                           If False, return a flattened list where each selected unit is
                           repeated according to its frequency.

        Behavior details:
            - `topk` controls how many distinct units are selected (not the length of the
              returned list). If fewer distinct units exist than `topk`, all are returned.
            - Ties are broken deterministically by unit string (ascending) after sorting by
              descending frequency.
            - When unique=True the returned list length is min(topk, number of distinct units).
            - When unique=False the returned list length equals the sum of counts for the
              selected top-k distinct units.

        Returns:
            List[str]: The selected units (unique or flattened by count).

        Raises:
            ValueError: If topk is not an int >= 1.
            TypeError: If text_units is not iterable or contains non-string elements.

        Examples:
            >>> tu = TextUnit()
            >>> text_units = ["apple","banana","apple","orange","banana","apple"]
            >>> tu.top_k_textunits(text_units, 2, unique=True)
            ['apple', 'banana']
            >>> tu.top_k_textunits(text_units, 2, unique=False)
            ['apple','apple','apple','banana','banana']
        """
        # Validation
        if not isinstance(topk, int) or topk < 1:
            raise ValueError("topk must be an integer >= 1.")

        try:
            # Ensure we can iterate and check element types
            iterator = iter(text_units)
        except TypeError:
            raise TypeError("text_units must be an iterable of strings.")

        # Build counter and ensure elements are strings
        counter = Counter()
        for x in text_units:
            if not isinstance(x, str):
                raise TypeError("All elements in text_units must be strings.")
            counter[x] += 1

        if not counter:
            return []

        # Deterministic tie-breaking: sort by (-count, unit)
        sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        selected = sorted_items[:topk]

        if unique:
            return [unit for unit, _count in selected]

        # Flatten selected units by their counts (preserves frequency ordering)
        return list(itertools.chain.from_iterable([unit] * count for unit, count in selected))
    
    
    def apostrophized_tokens(
        self,
        text: str,
        apostrophes: Optional[Iterable[str]] = None) -> List[Union[str, "spacy.tokens.token.Token"]]:
        """
        Return tokens that contain an apostrophe-like character.

        Scans the token sequence produced by self.tokens(text) and returns only
        those tokens whose text contains any character from `apostrophes`.
        The returned items preserve the type returned by self.tokens() (i.e.
        spaCy Token objects or plain strings), preserve input order, and keep
        duplicate occurrences.

        Parameters:
            text (str): Input text to tokenize and scan.
            apostrophes (Optional[Iterable[str]]): Iterable of characters to treat
                as apostrophes (e.g. {"'", "’", "‘"}). If None a sensible default
                set of apostrophe-like characters is used.

        Returns:
            List[Union[str, spacy.tokens.token.Token]]: Tokens containing any of the
                specified apostrophe characters.

        Raises:
            TypeError: If `text` is not a string or if elements of `apostrophes`
                are not single-character strings.

        Examples:
            >>> tu = TextUnit()
            >>> tu.apostrophized_tokens("John's book and o'clock")
            ["John's", "o'clock"]
        """
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")

        # default set of common apostrophe-like characters
        if apostrophes is None:
            apostrophes_set = {"'", "’", "‘", "ʼ", "ʹ"}
        else:
            # accept string or iterable; normalize to set of single-character strings
            apostrophes_set = set(apostrophes)
            if not all(isinstance(ch, str) and len(ch) >= 1 for ch in apostrophes_set):
                raise TypeError("`apostrophes` must be an iterable of one-or-more-character strings.")

        tokens = self.tokens(text)

        result: List[Union[str, "spacy.tokens.token.Token"]] = []
        for t in tokens:
            txt = t.text if hasattr(t, "text") else str(t)
            # check membership efficiently without building intermediate lists
            if any(ch in txt for ch in apostrophes_set):
                result.append(t)
        return result
    
    def contractions(self, text: str, full_form: bool = False) -> List[str]:
        """
        Extract contractions found in `text`.

        Behavior:
        - English: detects apostrophized tokens (e.g. "don't", "I'm") and returns them.
          If full_form=True the function attempts to expand each contraction
          (requires the third-party `contractions` package).
        - German: extracts alphabetic tokens and returns those that match the loaded
          contractions resource (expansion not implemented).
        - Preserves token order and duplicate occurrences.

        Parameters:
            text (str): Input text to scan for contractions.
            full_form (bool): If True, return expanded/full forms where supported
                (English only). Defaults to False.

        Returns:
            List[str]: List of found contractions (or their expanded forms when requested).

        Raises:
            TypeError: If `text` is not a string.
            RuntimeError: If full_form=True for English but the `contractions` package
                cannot be imported.
            NotImplementedError: If full_form=True for languages without expansion support.
            ValueError: If the TextUnit language is unsupported.
        """
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")

        # normalize common apostrophe-like characters to ASCII apostrophe for matching
        def _norm_apostrophes(s: str) -> str:
            return (
                s.replace("’", "'")
                 .replace("‘", "'")
                 .replace("ʼ", "'")
                 .replace("ʹ", "'")
                 .replace("´", "'"))

        if self.__language == self.Language.English:
            tokens = self.apostrophized_tokens(text)
            # Convert token objects to plain strings and normalize apostrophes
            token_texts = [_norm_apostrophes(t.text if hasattr(t, "text") else str(t)) for t in tokens]

            # Filter against contractions resource (resource entries are normalized already)
            matches = [t for t in token_texts if t.lower() in self.__contractions]

            if full_form:
                # Expand contractions using the `contractions` package if available
                try:
                    import contractions as _contractions_pkg  # type: ignore
                except Exception as e:
                    raise RuntimeError("contractions package is required to expand contractions (full_form=True).") from e

                # contractions.fix may expand single-token contractions to multi-word strings
                return [_contractions_pkg.fix(t) for t in matches]

            return matches

        elif self.__language == self.Language.German:
            # Use alphabetic tokens (fix previously misspelled method name)
            alpha_tokens = self.alphabetic_tokens(text=text, strategy=self.AlphaTokens.StripSurroundingPuncts, lowered=False)
            matches = [t for t in alpha_tokens if t.lower() in self.__contractions]

            if full_form:
                # Expansion for German contractions is not implemented
                raise NotImplementedError("Full-form contraction expansion for German is not implemented.")

            return matches

        else:
            raise ValueError(f"Unsupported language: {self.__language}")


    def hashtags(self, text: str) -> List[str]:
        """
        Extract social-media-style hashtags from text.

        This matches hashtags as used on platforms like X/Twitter: a '#' that is
        not part of an existing word, followed by one or more Unicode letters,
        marks, digits or underscore characters. Trailing punctuation is not
        included. The match preserves the leading '#' and original casing.

        Args:
            text (str): Input text to scan for hashtags.

        Returns:
            List[str]: All hashtags found in input text, in order of appearance.

        Raises:
            TypeError: If text is not a string.

        Examples:
            >>> tu = TextUnit()
            >>> tu.hashtags("Launching today: #ProductLaunch 🚀 #2025_goal #αβγ")
            ['#ProductLaunch', '#2025_goal', '#αβγ']
        """
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")

        # Pattern: ensure preceding char is not a word-like char (letter/mark/number/underscore)
        # then capture '#' plus one-or-more letters/marks/numbers/underscore.
        pattern = regex.compile(r"(?<![\p{L}\p{M}\p{N}_])(#(?:[\p{L}\p{M}\p{N}_]+))", flags=regex.VERSION1)
        return pattern.findall(text)


    def named_entities(self, 
                       text: str, 
                       restrict_to_categories: Optional[Iterable[str]] = None,
                       combine_with_tag: bool = False) -> List[Union[str, Tuple[str, str]]]:
        """
        Extract named entities from text using the spaCy pipeline.

        By default (combine_with_tag=False) returns entity surface forms (strings)
        in document order. If combine_with_tag=True returns a list of tuples
        (entity_text, entity_label). If restrict_to_categories is provided, only
        entities whose label matches one of the specified categories are returned;
        matching is case-insensitive and applies equally to both return formats.

        Parameters:
            text (str): Input text to analyze.
            restrict_to_categories (Optional[Iterable[str]]): Iterable of spaCy NER
                labels (e.g. ["PERSON", "ORG", "GPE"]). If None, all categories are returned.
            combine_with_tag (bool): If True, return (entity_text, entity_label) tuples.
                If False (default), return entity texts as strings.

        Returns:
            List[Union[str, Tuple[str, str]]]: Entities (strings) or (text,label) tuples
                in document order. Duplicate occurrences are preserved.

        Raises:
            TypeError: If `text` is not a string or `restrict_to_categories` is not iterable of strings.
            RuntimeError: If the spaCy pipeline is not initialized or does not include an NER component.

        Examples:
            >>> tu = TextUnit()
            >>> tu.named_entities("Barack Obama was born in Hawaii.")
            ['Barack Obama', 'Hawaii']
            >>> tu.named_entities("Barack Obama in Hawaii", restrict_to_categories=['GPE'], combine_with_tag=True)
            [('Hawaii', 'GPE')]
        """
        
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")

        if self.__nlp is None:
            raise RuntimeError("spaCy pipeline is not initialized.")

        if "ner" not in self.__nlp.pipe_names:
            raise RuntimeError("spaCy pipeline does not include an NER component ('ner').")

        categories: Optional[Set[str]] = None
        if restrict_to_categories is not None:
            try:
                categories = {str(c).upper() for c in restrict_to_categories}
            except TypeError:
                raise TypeError("`restrict_to_categories` must be an iterable of strings or string-like values.")

        doc = self.__nlp(text)
        out: List[Union[str, Tuple[str, str]]] = []

        if categories is None:
            if combine_with_tag:
                out = [(ent.text, ent.label_) for ent in doc.ents]
            else:
                out = [ent.text for ent in doc.ents]
            return out

        # Filter by label (preserve document order) and respect combine_with_tag
        if combine_with_tag:
            return [(ent.text, ent.label_) for ent in doc.ents if ent.label_.upper() in categories]
        return [ent.text for ent in doc.ents if ent.label_.upper() in categories]


    def lines(self, text: str, remove_empty_lines: bool = True) -> List[str]:
        """
        Split text into logical lines.

        Uses Python's str.splitlines() (which handles universal newlines) to
        split `text` into lines. By default empty/whitespace-only lines are
        removed; the returned lines do not include trailing newline characters.

        Parameters:
            text (str): Input text to split into lines.
            remove_empty_lines (bool): If True (default) filter out lines that
                are empty or contain only whitespace. If False, return all lines
                including those that are empty. Note: returned lines do not
                include the original newline characters.

        Returns:
            List[str]: List of lines in document order. Whitespace inside each
            line is preserved (only fully-empty lines may be removed).

        Raises:
            TypeError: If `text` is not a string.

        Examples:
            >>> tu = TextUnit()
            >>> tu.lines("a\\nb\\n\\n c\\n")
            ['a', 'b', ' c']
            >>> tu.lines("a\\r\\n\\nb", remove_empty_lines=False)
            ['a', '', 'b']

        Notes:
            - This function intentionally preserves leading/trailing whitespace
              of non-empty lines. If you need trimmed lines, call .strip() on
              each element of the returned list.
            - If you need to preserve newline characters, use text.splitlines(keepends=True)
              externally before calling this helper.
        """
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")

        lines = text.splitlines()
        if remove_empty_lines:
            return [s for s in lines if s.strip()]
        return lines


    def function_words(
        self,
        text: str,
        categories: Optional[List[Enum]] = None,
        lowered: bool = False) -> List[str]:
        """
        Extract function words from the given text.

        For English text function words can be filtered by category. If no
        categories are provided all available function words for English
        are used.

        For German text function word categories are not supported yet.
        In this case all available function words for German are used and
        the categories argument is ignored. A warning is issued to make
        this explicit.

        Parameters
        ----------
        text : str
            The input text from which function words are extracted.
        categories : list of Enum, optional
            Function word categories to be considered. Only supported for
            English. If None all function words of the active language are
            used.
        lowered : bool, optional
            If True the returned function words are lowercased.

        Returns
        -------
        List[str]
            A list of function words occurring in the text filtered by the
            chosen settings.
        """
        alpha_tokens = self.alphabetic_tokens(
            text=text,
            lowered=lowered,
            strategy=self.AlphaTokens.StripSurroundingPuncts)

        # Start from the full function word inventory for the current language
        considered_function_words: set[str] = set(self.__functionwords)

        # If the active language is German categories are not supported yet
        if self.language == self.Language.German:
            if categories is not None:
                warnings.warn(
                    "Function word categories are currently only supported for English. "
                    "For German text the 'categories' argument is ignored and all "
                    "available function words are used instead.",
                    UserWarning,
                    stacklevel=2)
            # considered_function_words already set to self.__functionwords
        else:
            # English or other languages where categories may exist
            if categories is not None:
                en_categories = {
                    self.FunctionWordCategoryEn.Conjunctions: self.en_functionwords_conjunctions,
                    self.FunctionWordCategoryEn.AuxiliaryVerbs: self.en_functionwords_auxiliary_verbs,
                    self.FunctionWordCategoryEn.Determiners: self.en_functionwords_determiners,
                    self.FunctionWordCategoryEn.Prepositions: self.en_functionwords_prepositions,
                    self.FunctionWordCategoryEn.Pronouns: self.en_functionwords_pronouns,
                    self.FunctionWordCategoryEn.Quantifiers: self.en_functionwords_quantifiers}

                considered_function_words = set()
                for category in categories:
                    if category in en_categories:
                        considered_function_words.update(en_categories[category])

        # Case insensitive lookup
        considered_function_words_lower = {fw.lower() for fw in considered_function_words}

        return [t for t in alpha_tokens if t.lower() in considered_function_words_lower]