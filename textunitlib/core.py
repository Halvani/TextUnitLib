from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, Iterable, List, Set, FrozenSet
from functools import lru_cache
from collections import Counter

import regex
import spacy
import emoji

#from nlp_handler import NlpHandler, SpacyModelSize 
from . import lib_functions
from .nlp_handler import NlpHandler, SpacyModelSize


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
        # misc
        resources_basepath: Union[str, Path] = "textunitlib/LinguisticResources",
        log_fn=None,
    ):
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
            self.__functionwords = self.__load_resource(base / "en" / "functionwords" / "all.txt")
        elif self.__language == self.Language.German:
            self.__vowels = self.__load_resource(base / "de" / "vowels" / "vowels.txt")
            self.__contractions = self.__load_resource(base / "de" / "contractions" / "contractions.txt")
            self.__stopwords = self.__load_resource(base / "de" / "stopwords" / "stopwords.txt")
            # If you have German function words, load them here:
            fw_all = base / "de" / "functionwords" / "all.txt"
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
            # Be lenient: return empty set but log would have helped already
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
            resulting_grams=resulting_grams,
        )
        
        
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
