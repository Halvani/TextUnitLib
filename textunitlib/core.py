import regex as re  # Difference between regex and re library: https://github.com/alvations/sacremoses/issues/76
import itertools
from functools import reduce
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import contractions
import fastnumbers
import torch
import spacy
import emoji
from typing import List, Set, Union
from enum import Enum, auto
import datefinder
from urlextract import URLExtract
from email_scraper import scrape_emails
from spacy.matcher import Matcher
from nltk.tokenize import wordpunct_tokenize


class TextUnit:
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

    # Private helper functions
    # ===========================================================================
    def __translate_language_to_code(self, language: Language) -> str:
        """
        Translate the given Language enum to its corresponding language code.

        Args:
            language (Language): The Language enum to be translated.

        Returns:
            str: The language code corresponding to the input Language enum.

        Raises:
            KeyError: If the provided Language enum is not found in the translation dictionary.
        """

        translation_dict = {
            TextUnit.Language.English: "en",
            TextUnit.Language.German: "de"
        }

        try:
            return translation_dict[language]
        except KeyError:
            raise KeyError(f"Language {language} is not supported.")

    def __none_or_empty(self, data: Union[str, List]) -> bool:
        """
        Check if the input data is None or empty.

        Args:
        - data: Any, the input data to be checked.

        Returns:
        - bool: True if the data is None or empty, False otherwise.

        Raises:
        - TypeError: If the input data is of an unsupported type.

        Example:
        >>> __none_or_empty(None)
        True

        >>> __none_or_empty("")
        True

        >>> __none_or_empty([])
        True

        >>> __none_or_empty("Hello")
        False

        >>> __none_or_empty([1, 2, 3])
        False
        """
        if data is None:
            return True

        elif isinstance(data, str):
            return not data.strip()

        elif isinstance(data, list):
            return not bool(data)

        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def __strip_surrounding_puncts(self, token: str, apostrophes=None) -> str:
        """
        Strips surrounding punctuation marks except inner hyphens and apostrophs from the given token.

        Args:
            token (str): The input token from which to strip surrounding punctuation.
            apostrophes (list, optional): A list of apostrophe characters to consider as valid. Defaults to ["’", "'"].

        Returns:
            str: The stripped token, excluding surrounding punctuation characters. If non-letter characters (except hyphen)
                 are found within the token, an empty string is returned, indicating that stripping is not possible.
        
        Example:
        >>> __strip_surrounding_puncts("  can't  ")
        "can't"
        >>> __strip_surrounding_puncts("**Anna-Lena++#")
        "Anna-Lena"
        >>> __strip_surrounding_puncts("jones'")
        "jones'"
        """

        if self.__none_or_empty(apostrophes):
            apostrophes = apostrophes = ["’", "'"]

        stripped_left = "".join(itertools.dropwhile(lambda x: not x.isalpha(), token))
        stripped_right = "".join(
            itertools.dropwhile(lambda x: not (x.isalpha() or x in apostrophes), stripped_left[::-1]))
        stripped = stripped_right[::-1]

        valid_chars = apostrophes + ["-"]
        bin_string = "".join(map(lambda c: str(int(c.isalpha() or c in valid_chars)), stripped))
        first, last = bin_string.find("1"), bin_string.rfind("1")

        # If a non-letter (except hyphen) is inbetween the string we cannot strip it. In this case, we return
        # an empty string and perform the respective handling when calling the function.  
        return "" if "0" in bin_string[first:last] else stripped[first:last + 1]

    def __load_resource(self, filepath_resource: str = None, load_as_frozenset: bool = True):
        """
        Load a linguistic resource from the specified file path and return its content.

        Args:
        - filepath_resource (str): The file path of the linguistic resource to be loaded.
        - load_as_frozenset (bool): If True, the content will be returned as a list of strings.
                             If False, the content will be returned as a frozenset of strings.

        Returns:
        - list or frozenset: The content of the loaded linguistic resource.

        Raises:
        - FileNotFoundError: If the specified file path does not exist.

        Example:
        >>> __load_resource(r"LinguisticResources/independent/apostrophes.txt", load_as_frozenset=True)
        {"’", "'"}
        """
        if not Path(filepath_resource).exists:
            raise FileNotFoundError(
                f"Resource could not be loaded. The specified file: {filepath_resource} does not exist.")

        list_of_strings = Path(filepath_resource).read_text(encoding="utf8").splitlines()

        if load_as_frozenset:
            return frozenset(list_of_strings)

        return list_of_strings

    def __sort_by_alphabet_thenby_len(self, list_of_text_units: List[str]) -> List[str]:
        """
        Sorts a list of text units alphabetically and then by length.

        Parameters:
        - text_units (List[str]): A list of strings representing text units to be sorted.

        Returns:
        - List[str]: A new list containing the elements from the input list,
             sorted first alphabetically and then by length.

        Example:
        >>> text_units = ["Ant", "Aardvark", "Chihuahua", "Bear", "Bee", "Camel"]
        >>> print(__sort_by_alphabet_thenby_len(text_units))
        ["Aardvark", "Ant", "Bear", "Bee", "Camel", "Chihuahua"]
        """
        return sorted(list_of_text_units, key=lambda x: (x, len))

    def __get_maximum_substrings(self, list_of_text_units: List[str]) -> Set[str]:
        """
        Finds and returns the maximum substrings within a given list of text units.

        Parameters:
        - list_of_text_units (List[str]): A list of text units to find maximum substrings within.

        Returns:
        - Set[str]: A set containing the maximum substrings found within the given list.

        Note:
        The maximum substring is defined as the longest string within the list
        that contains another string within it. The function sorts the input list
        by alphabet and length and iterates through it to identify and collect
        maximum substrings.

        Example:
        >>> text_units = ["A", "AB", "ABCD", "X", "XXX", "ABC", "Y", "YY"]
        >>> print(__get_maximum_substrings(text_units))
        {"ABCD", "XXX", "YY"}
        """
        sorted_text_units = self.__sort_by_alphabet_thenby_len(list_of_text_units)
        max_substrings = set()
        for i in range(0, len(sorted_text_units)):
            for j in range(len(sorted_text_units) - 1, -1, -1):
                if sorted_text_units[i] in sorted_text_units[j]:
                    max_substrings.add(sorted_text_units[j])
                    break
        return max_substrings

    def __extract_integers(self, list_of_text_units: List[str]):
        """ TODO: Description 
        Note, to allow language support for e.g. seperators, use locale --> https://fastnumbers.readthedocs.io/en/2.0.1/intro.html
        import locale
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
        print(atof('468,5', func=fast_float))  # Prints 468.5
        """
        return [t for t in list_of_text_units if fastnumbers.isint(t)]

    def __extract_floats(self, list_of_text_units: List[str], cover_integers: bool = False):
        """ TODO: Description 
        Note, to allow language support for e.g. seperators, use locale --> https://fastnumbers.readthedocs.io/en/2.0.1/intro.html
        import locale
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
        print(atof('468,5', func=fast_float))  # Prints 468.5
        """
        if cover_integers:
            return [t for t in list_of_text_units if fastnumbers.isfloat(t)]

        integers = set(self.__extract_integers(list_of_text_units))
        return [t for t in list_of_text_units if t not in integers and fastnumbers.isfloat(t)]

    def __extract_spelled_out_numbers(self, text):
        """ TODO: Description 
        Check out --> https://github.com/allo-media/text2num Perhaps better than the following solution.
        """

        matcher = Matcher(self.__nlp.vocab)

        pattern = [{'POS': 'NUM', 'OP': '+'},
                   {'TEXT': '-', 'OP': '?'},
                   {'POS': 'NUM', 'OP': '*'}]

        matcher.add("SpelledOutNumber", [pattern])
        doc = self.__nlp(text)
        matches = matcher(doc)

        spans = [doc[start:end] for _, start, end in matches]
        result = []
        for span in spacy.util.filter_spans(spans):
            result.append(span.text)

        result = [n for n in result if not n.isdigit() or not n.isdecimal() or not n.isnumeric()]

        return result
    # ===========================================================================

    def __init__(self, language: Language = None, nlp=None):
        """ TODO: Description """

        # The folder which contains hard coded linguistic resources.
        self.__ling_res_basepath = r"textunitlib/LinguisticResources"

        # If no language is specified, English is chosen by default.
        self.__language = self.Language.English if language is None else language

        if nlp is None or not isinstance(nlp, spacy.Language):
            # TODO: Use English model by default !
            raise AssertionError(
                "To create a TextUnitLib instance, a basic spaCy nlp pipeline must be provided beforehand.")
        else:
            self.__nlp = nlp
            self.__original_nlp_ruleset = nlp.tokenizer.rules

        # Load linguistic resources according to selected language.
        if language == self.Language.English:
            self.__vowels = self.__load_resource(f"{self.__ling_res_basepath}/en/vowels/vowels.txt")
            self.__contractions = self.__load_resource(f"{self.__ling_res_basepath}/en/contractions/contractions.txt")
            self.__stopwords = self.__load_resource(f"{self.__ling_res_basepath}/en/stopwords/stopwords.txt")
            self.__functionwords = self.__load_resource(f"{self.__ling_res_basepath}/en/functionwords/all.txt")

        #         self.en_functionwords_conjunctions = frozenset(Path(f"{self.en_functionwords_path}/conjunctions.txt")
        #                                                  .read_text(encoding="utf8").splitlines())
        #         self.en_functionwords_auxiliary_verbs = frozenset(Path(f"{self.en_functionwords_path}/auxiliary_verbs.txt")
        #                                                     .read_text(encoding="utf8").splitlines())
        #         self.en_functionwords_determiners = frozenset(Path(f"{self.en_functionwords_path}/determiners.txt")
        #                                                 .read_text(encoding="utf8").splitlines())
        #         self.en_functionwords_prepositions = frozenset(Path(f"{self.en_functionwords_path}/prepositions.txt")
        #                                                  .read_text(encoding="utf8").splitlines())
        #         self.en_functionwords_pronouns = frozenset(Path(f"{self.en_functionwords_path}/pronouns.txt")
        #                                              .read_text(encoding="utf8").splitlines())
        #         self.en_functionwords_quantifiers = frozenset(Path(f"{self.en_functionwords_path}/quantifiers.txt")
        #                                                 .read_text(encoding="utf8").splitlines())
        #         self.functionwords_en = frozenset(self.en_functionwords_conjunctions |
        #                                           self.en_functionwords_auxiliary_verbs |
        #                                           self.en_functionwords_determiners |
        #                                           self.en_functionwords_prepositions |
        #                                           self.en_functionwords_pronouns |
        #                                           self.en_functionwords_quantifiers)

        elif language == self.Language.German:
            self.__vowels = self.__load_resource(f"{self.__ling_res_basepath}/de/vowels/vowels.txt")
            self.__contractions = self.__load_resource(f"{self.__ling_res_basepath}/de/contractions/contractions.txt")
            self.__stopwords = self.__load_resource(f"{self.__ling_res_basepath}/de/stopwords/stopwords.txt")

        # Unsupported language
        else:
            raise self.Language.UnsupportedLanguage("TBD...")

        # Punctuation marks        
        self.__punctuation = self.__load_resource(f"{self.__ling_res_basepath}/independent/punctuation.txt")

    def textunit_ngrams(self,
                        text_units: Union[str, List[str]],
                        n: int,
                        sep: str = " ",
                        strip_spaces: bool = False) -> List[str]:
        """
        Generate n-grams from a given list of text units or a string.

        Parameters:
        - text_units (Union[str, List[str]]): Input text units. If a string is provided,
        it will be treated as individual characters.
        - n (int): Size of the n-grams to generate.
        - sep (str, optional): Separator between characters in an n-gram. Default is a space.
        - strip_spaces (bool, optional): If True, strip leading and trailing spaces from each n-gram. Default is False.

        Returns:
        List[str]: A list of n-grams generated from the input text units.

        Raises:
        - ValueError: If the specified n-gram size is larger than the number of text units.

        Example:
        >>> textunit_ngrams("TextUnit", n=4)
        ['Text', 'extU', 'xtUn', 'tUni', 'Unit']

        >>> textunit_ngrams(["Text", "Unit", "Lib"], n=2, sep="_")
        ['Text_Unit', 'Unit_Lib']

        >> textunit_ngrams([" Text", "Unit", "Lib "], n=2, strip_spaces=True)
        ['Text Unit', 'Unit Lib']
        """

        if isinstance(text_units, str):
            sep = ""
            text_units = list(text_units)

        if n > len(text_units):
            raise ValueError("Cannot generate ngrams of a size bigger than the number of text units!")

        if n == 1:
            return text_units if not strip_spaces else [t.strip() for t in text_units]
        else:
            ngrams = [sep.join(text_units[i:i + n]) for i in range(len(text_units) - n + 1)]

        if strip_spaces:
            return [n.strip() for n in ngrams]

        return ngrams

    def characters(self, text: str, drop_whitespaces: bool = False) -> List[str]:
        """
        Extracts characters from the input text, optionally excluding spaces.

        Args:
        - text (str): The input text from which characters will be extracted.
        - drop_whitespaces (bool, optional): If True, spaces will be excluded from the result.

        Returns:
        List[str]: A list of characters extracted from the input text.

        Examples:
        >>> characters(" Text Unit Lib ")
        [' ', 'T', 'e', 'x', 't', ' ', 'U', 'n', 'i', 't', ' ', 'L', 'i', 'b', ' ']

        >>> characters(" Text Unit Lib ", drop_whitespaces=True)
        ['T', 'e', 'x', 't', 'U', 'n', 'i', 't', 'L', 'i', 'b']
        """

        return [c for c in text if not (drop_whitespaces and c.isspace())]

    def spaces(self, text: str) -> List[str]:
        """Return a list of whitespace characters found in the given text.

        This function takes a text input and identifies and returns a list
        containing all whitespace characters present in the input text.

        Args:
            text (str): The input text to analyze for whitespace characters.

        Returns:
            List[str]: A list of whitespace characters found in the input text.

        Example:
            >>> spaces(" Text	Unit 	Lib	 ")
            [' ', '\t', ' ', '\t', '\t', ' ']

        Note:
            This function utilizes the 'characters' method with 'drop_whitespaces' set to False
            in order to extract all characters, and then filters out non-whitespace characters.
        """

        chars = self.characters(text, drop_whitespaces=False)
        return [s for s in chars if s.isspace()]

    def punctuation_marks(self, text: str) -> List[str]:
        """
        Extracts punctuation marks from the given text.

        Args:
        - text (str): The input text from which punctuation marks are extracted.

        Returns:
        - List[str]: A list containing punctuation marks found in the text.

        Example:
        >>> punctuation_marks("Hello, TextUnitLib! How are you?")
        [',', '!', '?']

        Note:
        - The function relies on the `characters` method of the current instance (`self`) to obtain a list of characters from the text.
        - The list of punctuation marks is defined in the `__punctuation` attribute of the current instance (`self`).
        """

        chars = self.characters(text)
        return [p for p in chars if p in self.__punctuation]

    def vowels(self, text: str) -> List[str]:
        """
        Extracts language-specific vowels from the given text.

        Args:
        - text (str): The input text from which vowels will be extracted.

        Returns:
        - List[str]: A list of all vowels that occur in the text in the respective language.

        Example:
        >>> vowels("Hello, how are you?")
        ['e', 'o', 'o', 'a', 'e', 'y', 'o', 'u']
        """

        chars = self.characters(text)
        return [v for v in chars if v.lower() in self.__vowels]

    def letters(self, text: str) -> List[str]:
        """
        Extracts and returns a list of alphabetical characters from the input text.

        Args:
        - text (str): The input text from which alphabetical characters will be extracted.

        Returns:
        - List[str]: A list containing only alphabetical characters from the input text in the respective language.

        Example:
        >>> letters("TextUnitLib --> Released @2024!")
        ['T', 'e', 'x', 't', 'U', 'n', 'i', 't', 'L', 'i', 'b', 'R', 'e', 'l', 'e', 'a', 's', 'e', 'd']
        """
        return [c for c in text if c.isalpha()]

    def digits(self, text: str) -> List[str]:
        """
        Extracts and returns a list of digit characters from the given text.

        Args:
        - text (str): The input text containing alphanumeric characters.

        Returns:
        - List[str]: A list of digits found in the input text.

        Example:
        >>> digits("abc123xyz456")
        ['1', '2', '3', '4', '5', '6']
        """
        return [c for c in text if c.isdigit()]

    def char_ngrams(self, text: str, n: int, strip_spaces: bool = False) -> List[str]:
        """Generate character n-grams from the input text.

        This function takes a text input and produces a list of character n-grams
        with a specified length 'n'. Optionally, leading and trailing spaces can be
        stripped from each n-gram if 'strip_spaces' is set to True.

        Args:
            text (str): The input text from which n-grams will be generated.
            n (int): The length of the desired character n-grams.
            strip_spaces (bool, optional): If True, leading and trailing spaces
                will be stripped from each generated n-gram. Default is False.

        Returns:
            List[str]: A list containing character n-grams based on the input text.

        Example:
            >>> char_ngrams("Hello World!", 3)
            ['Hel', 'ell', 'llo', 'lo ', 'o W', ' Wo', 'Wor', 'orl', 'rld', 'ld!']

            >>> char_ngrams("Hello World!", 3, strip_spaces=True)
            ['Hel', 'ell', 'llo', 'lo', 'o W', 'Wo', 'Wor', 'orl', 'rld', 'ld!']
        """

        return self.textunit_ngrams(text, n=n, strip_spaces=strip_spaces)

    def char_ngrams_range(self, text: str, n_from: int = 3, n_to: int = 5) -> List[str]:
        """
        Generate character n-grams within a specified range from the given text.

        Args:
        - text (str): The input text from which n-grams will be generated.
        - n_from (int, optional): The minimum length of the n-grams (default is 3).
        - n_to (int, optional): The maximum length of the n-grams (default is 5).

        Returns:
        list: A list containing all the generated n-grams within the specified range.

        Example:
        >>> char_ngrams_range("example text")
        ["exa", "xam", "amp", "mpl", "ple", "le ", "e t", " te", "tex", "ext", "exam", "xamp", "ampl", "mple", "ple ",
        "le t", "e te", " tex", "text", "examp", "xampl", "ample", "mple ", "ple t", "le te", "e tex", " text"]
        """
        range_ngrams = []
        for r in range(n_from, n_to + 1):
            current_ngrams = [text[i:i + r] for i in range(len(text.rstrip()) - r + 1)]
            range_ngrams.extend(current_ngrams)
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

        graphemes = re.findall(r"\X", text)
        emojis = [grapheme for grapheme in graphemes if emoji.is_emoji(grapheme)]
        return [emoji.demojize(c, language=lang_code) for c in emojis] if demojize else emojis

    def regex_strings(self, text: str, pattern: str, flags=0) -> List[str]:
        """ Flags --> https://docs.python.org/2.7/howto/regex.html#compilation-flags """
        if self.__none_or_empty(pattern):
            return list()
        return re.findall(f"{pattern}", text, flags=flags)

    def tokens(self, text: str, strategy=Tokenization.Whitespace, sep: str = None, ) -> List[str]:
        """ TODO: Description """

        if strategy == self.Tokenization.Whitespace:
            return text.split() if sep is None else text.split(sep=sep)

        elif strategy == self.Tokenization.WhitespacePunctuation:
            return wordpunct_tokenize(text)

        elif strategy == self.Tokenization.SpacyTokens:
            return [t for t in self.__nlp(text)]

        elif strategy == self.Tokenization.SpacyStringTokens:
            return [t.text for t in self.__nlp(text)]

    def alphabethic_tokens(self,
                           text: str,
                           lowered: bool = False,
                           strategy=AlphaTokens.StripSurroundingPuncts,
                           min_length: int = 1) -> List[str]:
        """
        Extracts all tokens in a text which consist of the alphabet of the respective language.
        Note, alphabetic tokens do not necessarily constitute words. However, the opposite is true.
        
        Parameters:
           text (str): A text containing some characters.
           
           lowered (bool): Decide whether the extracted tokens should be lowercased or not.
           
           strategy: A strategy that specifies the extent to which alphabetic letters should be considered within the extracted tokens.  
           AlphaTokens.StripSurroundingPuncts: (Default) Strips surrounding punctuation marks from each token T.
           AlphaTokens.OnlyAlphaLetters: Considers each token T as a whole if T consists only of letters (+ hyphen). 
           AlphaTokens.Greedy: Selects letters distributed over the given text, accepting hyphens within but not 
           at the beginning or end of the text. For example, T = "#+?=twe__nty-on//e!!?§%" results in "twenty-one".
           
           min_length (int): The minimum accepted length of each extracted token T.

        Returns:
            List[str]: A list of tokens consisting solely of letters as well as (optional) hyphens and apostrophes. 
        """
        tokens = self.tokens(text)

        alpha_tokens = []
        for token in tokens:
            if strategy == self.AlphaTokens.StripSurroundingPuncts:
                cleaned_token = self.__strip_surrounding_puncts(token)
                if len(cleaned_token.strip()) > 0 and not (cleaned_token.startswith("-") and
                                                           cleaned_token.endswith("-") and
                                                           cleaned_token.startswith("'") and
                                                           cleaned_token.endswith("'")):
                    alpha_tokens.append(cleaned_token.lower() if lowered else cleaned_token)

            elif strategy == self.AlphaTokens.OnlyAlphaLetters:
                # Extend isalpha() to also allow hyphens, which are otherwise not included.
                if token.replace("-", "").replace("'", "").isalpha():
                    alpha_tokens.append(token.lower() if lowered else token)

            elif strategy == self.AlphaTokens.Greedy:
                candidate = "".join([c for c in token if c.isalpha() or c == "-" or c == "'"]).strip("-")
                if len(candidate) >= min_length:
                    alpha_tokens.append(candidate.lower() if lowered else candidate)
                else:
                    continue
        return alpha_tokens

    def vocabulary(self, text: str, strategy=AlphaTokens.StripSurroundingPuncts):
        """ TODO: Description """
        alpha_tokens = self.alphabethic_tokens(text, lowered=True, strategy=strategy)
        return sorted(set(alpha_tokens))

    def apostrophized_tokens(self, text: str, apostrophes: Set[str] = None):
        """ TODO: Description """

        # https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html
        tokens = self.tokens(text)

        if apostrophes is None:
            apostrophes = set("\'‘’")

        apostrophized_tokens = []
        for t in tokens:
            if any([c in t for c in apostrophes]):
                apostrophized_tokens.append(t)
        return apostrophized_tokens

    def contractions(self, text: str, full_form: bool = False):
        """
        Extracts contractions from the given text based on the specified language.

        Args:
            text (str): The input text from which contractions are to be extracted.
            full_form (bool, optional): If True, returns the full forms of contractions using contractions.fix().
                                         Defaults to False.

        Returns:
            List[str]: A list of extracted contractions or their full forms if full_form is True.

        Raises:
            Language.UnsupportedLanguage: If the specified language is not supported.
        """

        # Note, in contrast to languages such as German, English contractions represent apostrophized words. 
        # Hence, an individual handling is needed here. 
        if self.__language == self.Language.English:
            apostrophized_tokens = self.apostrophized_tokens(text)
            extracted_contractions = [t for t in apostrophized_tokens if
                                      t.replace("’", "'").lower() in self.__contractions]

            if full_form:
                return [contractions.fix(c) for c in extracted_contractions]
            return extracted_contractions

        elif self.__language == self.Language.German:
            alpha_tokens = self.alphabethic_tokens(text=text, strategy=self.AlphaTokens.StripSurroundingPuncts)
            extracted_contractions = [t for t in alpha_tokens if t.lower() in self.__contractions]

            # TODO: contraction fix for German --> use pikled dictionary for this purpose!
            # ---------------------------------------------------------------------------
            # if full_form:
            #                 return [contractions.fix(c) for c in extracted_contractions]

            return extracted_contractions

        else:
            raise self.Language.UnsupportedLanguage("TBD...")

    def postags(self,
                text: str,
                postag_type=PostagType.Universal,
                combine_with_token: bool = False,
                combine_sep: str = None,
                tags_to_consider: List[str] = None):
        """ TODO: Description """

        result = []

        spacy_tokens = self.tokens(text, strategy=self.Tokenization.SpacyTokens)
        for t in spacy_tokens:
            token = t.text
            postag = t.pos_ if postag_type == self.PostagType.Universal else t.tag_

            # Skip if the current postag is not within the given postag set.
            if tags_to_consider is not None and postag not in tags_to_consider:
                continue

            # Take all postags into account and process remaining conditions.
            else:
                # Consider tokens + postags.
                if combine_with_token:
                    # Append a (token, postag) tuple if a valid separator is given.
                    if self.__none_or_empty(combine_sep):
                        result.append((token, postag))
                        # Otherwise, append "token <sep> postag" as a single string.
                    else:
                        result.append(f"{token}{combine_sep}{postag}")
                        # Consider only postags.
                else:
                    result.append(postag)
        return result

    def postag_ngrams(self,
                      text: str,
                      n: int,
                      postag_type=PostagType.Universal,
                      combine_with_token: bool = False,
                      combine_sep: str = None,
                      tags_to_consider: List[str] = None,
                      ngram_sep: str = " "):
        """ TODO: Description """
        postags = self.postags(text=text,
                               postag_type=postag_type,
                               combine_with_token=combine_with_token,
                               combine_sep=combine_sep,
                               tags_to_consider=tags_to_consider)

        return self.textunit_ngrams(text_units=postags, n=n, sep=ngram_sep)

    def words(self, text: str, heuristic=WordHeuristic.AlphaTokens) -> List[str]:
        """ https://www.quora.com/What-is-the-difference-between-word-and-lexeme
        TODO: Integrate the following heuristics ["non-gibberish", "alpha"]
        """

        if heuristic == self.WordHeuristic.AlphaTokens:
            words = self.alphabethic_tokens(text=text, strategy=self.AlphaTokens.StripSurroundingPuncts)

        elif heuristic == self.WordHeuristic.Postags:
            # Note, since postags might even fail on punctuation (predicting them as word forms such as ADV)
            # we perform an extra check of weather the words solely consists of alphabetic letters.
            # However, this leads to ignoring apostrophe strings such as ["ca", "n't"]
            # To overcome this issue, we redefine the rules of the nlp model...

            self.__nlp.tokenizer.rules = {key: value for key, value in self.__nlp.tokenizer.rules.items() if
                                          "'" not in key and "’" not in key and "‘" not in key}
            non_word_tags = set(["PUNCT", "SYM", "SPACE", "X"])

            token_and_postags = self.universal_postags(text, combine_with_token=True)
            words = [w for w, p in token_and_postags if (w.isalpha() and p not in non_word_tags)]

            # Restore spaCy's default tokenizer rules, as otherwise the default models (tagger, parser, NER)
            # provided by spacy for English won't work as well on texts with this tokenization because they're
            # trained on data with the contractions split. Credits to: aab --> https://stackoverflow.com/a/59582203
            self.__nlp.tokenizer.rules = self.original_nlp_ruleset

        elif heuristic == self.WordHeuristic.NonGibberish:
            raise NotImplementedError("TBD...")

        return words

    def legomenon_units(self, text_units: List[str], n: int) -> List[str]:
        """
        Identify and return text elements that occur exactly 'n' times in the input list.

        Parameters:
        - text_units (List[str]): A list of text units (e.g., words, postags, or phrases) to analyze.
        - n (int): The target frequency of occurrences for identifying text units.

        Returns:
        - List[str]: A list of text units that occur exactly 'n' times in the input list.

        Example:
        >>> text_units = ["apple", "banana", "apple", "orange", "banana", "apple"]
        >>> legomenon_units(text_units, 2)
        ["banana"]
        """
        units_by_frequencies = Counter(text_units)
        return [t for t, freq in units_by_frequencies.items() if freq == n]

    def hapax_legomenon_units(self, text_units: List[str]) -> List[str]:
        """ Returns arbitrary text units that appear only *once* in a text """
        return self.legomenon_units(text_units, 1)

    def dis_legomenon_units(self, text_units: List[str]) -> List[str]:
        """ Returns arbitrary text units that appear only *twice* in a text """
        return self.legomenon_units(text_units, 2)

    def tris_legomenon_units(self, text_units: List[str]) -> List[str]:
        """ Returns arbitrary text units that appear only *three times* in a text """
        return self.legomenon_units(text_units, 3)

    def token_ngrams(self, text: str, n: int = 3, sep: str = " ", strip_spaces: bool = False) -> List[str]:
        """
        Given a text, return a list of arbitrary text units of a window size n (where n is a positive integer) 
        that overlap by a single unit.
        """
        return self.textunit_ngrams(self.tokens(text), n=n, sep=sep, strip_spaces=strip_spaces)

    def top_k_textunits(self, text_units, topk, unique=False) -> List[str]:
        """
        Retrieve the top k text units based on their frequencies in the input list.

        Parameters:
        - text_units (List[str]): A list of text units to analyze.
        - topk (int): The number of top text units to retrieve.
        - unique (bool, optional): If True, returns only unique text units. 
          Defaults to False, allowing repeated text units in the result.

        Returns:
        List[str]: A list containing the top k text units based on their frequencies.
                  If 'unique' is True, the list contains only the unique text units.

        Example:
        >>> text_units = ["apple", "banana", "apple", "orange", "banana", "apple"]
        >>> top_k_textunits(text_units, 2)
        ['apple', 'banana']

        >>> top_k_textunits(text_units, 2, unique=True)
        ['apple', 'banana']
        """
        topk_textunit_frequencies = Counter(text_units).most_common(topk)
        if unique:
            return list(map(lambda x: x[0], topk_textunit_frequencies))

        repetitions = [[x[0]] * x[1] for x in topk_textunit_frequencies]
        return list(itertools.chain(*repetitions))

    def lemmas(self, text: str) -> List[str]:
        """ TODO: Description """
        spacy_tokens = self.tokens(text, strategy=self.Tokenization.SpacyTokens)
        return [token.lemma_ for token in spacy_tokens]

    def hashtags(self, text: str) -> List[str]:
        """ TODO: Description """
        return re.findall("(#[a-zA-Z0-9(_)]+)", text)

    def named_entities(self, text: str, restrict_to_categories: List[str] = None) -> List[str]:
        """ TODO: Description """
        all_named_entities = {ner_tag: [] for ner_tag in self.nlp.pipe_labels["ner"]}

        doc = self.__nlp(text)
        for word in doc.ents:
            all_named_entities[word.label_].extend([word.text])

        # Return all named entities when no restriction is made.
        if restrict_to_categories is None or len(restrict_to_categories) == 0:
            return [element for sublist in all_named_entities.values() for element in sublist]
        else:
            temp = [category_entries for (ne_category, category_entries) in all_named_entities.items() if
                    ne_category in set(restrict_to_categories)]
            return [element for sublist in temp for element in sublist]

    def sentences(self, text: str, remove_empty_lines: bool = True) -> List[str]:
        """ TODO: Description """
        doc = self.__nlp(text)
        sentences = list(doc.sents)
        if remove_empty_lines:
            return [s.text for s in sentences if len(s.text.strip()) > 0]
        return sentences

    def lines(self, text: str, remove_empty_lines: bool = True) -> List[str]:
        """ TODO: Description """
        lines = text.splitlines()
        if remove_empty_lines:
            return [s for s in lines if len(s.strip()) > 0]
        return lines

    def textunit_lengths(self,
                         text_units: List[str],
                         lengths_only: bool = False,
                         sort_by_length: bool = False) -> List[str]:
        """ Given a list of arbitrary text units (e.g., letters, words or phrases) 
        return either a list of their lengths or a dictionary of the form length --> [text units])"""

        if lengths_only:
            return [len(u) for u in text_units]

        if sort_by_length:
            text_units = sorted(text_units, key=lambda x: len(x))

        length_distribution = defaultdict(list)
        for text_unit in text_units:
            length_distribution[len(text_unit)].append(text_unit)
        return length_distribution

    def gibberish_tokens(self, text: str) -> List[str]:
        raise NotImplementedError("TBD...")

    def homophones(self, text: str) -> List[str]:
        raise NotImplementedError("TBD...")

    #         postags_and_tokens = self.universal_postags(text, combine_with_token=True)

    def quotes(self, text: str) -> List[str]:
        raise NotImplementedError("TBD...")

    def rare_tokens(self, text: str, information_content_threshold: float = 3.0) -> List[str]:
        # TODO: First generate tokens and based on these compute their information content using the gibberish_tokens()
        raise NotImplementedError("TBD...")

    #     def homonyms(self, text: str):
    #         postags_and_tokens = self.universal_postags(text, combine_with_token=True)

    def stop_words(self, text: str, lowered: bool = False) -> List[str]:
        """
        Extracts stop words from the given text.

        Parameters:
        - text (str): The input text from which stop words will be extracted.
        - lowered (bool, optional): If True, performs case-insensitive matching for stop words. Defaults to False.

        Returns:
        - list: A list of stop words found in the input text.

        Note:
        The function uses the `alphabethic_tokens` method with the specified tokenization strategy
        to extract alphabetic tokens from the text. Stop words are then filtered based on case sensitivity.

        Example:

        >>> stop_words("This is a sample sentence with some common stop words.")
        ['is', 'a', 'with', 'some']

        >>> stop_words("Another example with mixed case stop words.", lowered=True)
        ['Another', 'example', 'with', 'mixed', 'case']
        """
        alpha_tokens = self.alphabethic_tokens(text=text, lowered=lowered,
                                               strategy=self.AlphaTokens.StripSurroundingPuncts)
        stopwords = ([a for a in alpha_tokens if a in self.__stopwords] if lowered else
                     [a for a in alpha_tokens if a.lower() in self.__stopwords])

        return stopwords

    def function_words(self, text: str, categories: List[Enum] = None, lowered=False) -> List[str]:
        """ TODO: Description; Integration for German. """

        alpha_tokens = self.alphabethic_tokens(text=text, lowered=lowered,
                                               strategy=self.AlphaTokens.StripSurroundingPuncts)
        considered_function_words = set()

        # If no categories specified use all available function words by default.
        if categories is None:
            considered_function_words = self.__functionwords
        else:
            en_categories = OrderedDict(
                [(self.FunctionWordCategoryEn.Conjunctions, self.en_functionwords_conjunctions),
                 (self.FunctionWordCategoryEn.AuxiliaryVerbs, self.en_functionwords_auxiliary_verbs),
                 (self.FunctionWordCategoryEn.Determiners, self.en_functionwords_determiners),
                 (self.FunctionWordCategoryEn.Prepositions, self.en_functionwords_prepositions),
                 (self.FunctionWordCategoryEn.Pronouns, self.en_functionwords_pronouns),
                 (self.FunctionWordCategoryEn.Quantifiers, self.en_functionwords_quantifiers)])

            for category in categories:
                if category in en_categories:
                    considered_function_words = considered_function_words.union(en_categories[category])

        return [t for t in alpha_tokens if t in considered_function_words]

    def function_word_phrases(self, text: str, n: int = 3, keep_only_longest_phrases=True, lowered=False) -> List[str]:
        """
        Extracts function word phrases from the given text.

        Function words are words that serve a grammatical purpose rather than
        conveying meaningful content. This function tokenizes the input text,
        generates n-grams of specified lengths, and identifies phrases composed
        entirely of function words.

        Parameters:
        - text (str): The input text from which function word phrases are extracted.
        - n (int, optional): The maximum length of n-grams to consider (default is 3).
        - keep_only_longest_phrases (bool, optional): If True, keeps only the longest
          function word phrases; otherwise, returns all identified phrases (default is True).
        - lowered (bool, optional): If True, converts the input text to lowercase before processing
          (default is False).

        Returns:
        - List[str]: A list of function word phrases extracted from the input text.
          If keep_only_longest_phrases is True, it contains only the longest phrases.

        Note:
        The function uses an internal list of function words stored in self.__functionwords.
        """
        alpha_tokens = self.alphabethic_tokens(text, lowered=lowered, strategy=self.AlphaTokens.StripSurroundingPuncts)
        all_ngrams_in_range = []

        for i in range(1, n + 1):
            all_ngrams_in_range.extend(self.textunit_ngrams(alpha_tokens, n=i, sep=" ", strip_spaces=False))

        function_word_phrases = []
        for ngram in all_ngrams_in_range:
            ngram_tokens = ngram.lower().split()

            if all(map(lambda x: x in self.__functionwords, ngram_tokens)):
                function_word_phrases.append(ngram)

        return (self.__get_maximum_substrings(function_word_phrases)
                if keep_only_longest_phrases else function_word_phrases)

    def numerals(self, text: str,
                 numeral_type: NumeralType = NumeralType.Numerals,
                 allow_decimal_separators: bool = True,
                 decimal_separators: List[str] = None) -> List[str]:
        """ TODO: Description """
        # By definition: isdecimal() ⊆ isdigit() ⊆ isnumeric() -->
        # https://stackoverflow.com/questions/44891070/whats-the-difference-between-str-isdigit-isnumeric-and-isdecimal-in-pyth
        # Common decimal_separators --> https://en.wikipedia.org/wiki/Decimal_separator

        # str.isdecimal() --> Only decimal numbers
        # str.isdigit() --> Decimals, subscripts, superscripts
        # str.isnumeric() --> Decimals, subscripts, superscripts, vulgar fractions, roman numerals, currency numerators

        decimal_separators = [",", "."] if decimal_separators is None or len(
            decimal_separators) == 0 else decimal_separators
        replacements = {d: "" for d in decimal_separators}
        handle_separators = lambda t: reduce(lambda x, kv: x.replace(*kv), replacements.items(),
                                             t) if allow_decimal_separators else t

        if numeral_type == self.NumeralType.Decimals_0_to_9:
            tokens = self.tokens(text, strategy=self.Tokenization.WhitespacePunctuation)
            zero_to_nine_digits = set("0123456789")
            return [t for t in tokens if all(c in zero_to_nine_digits for c in t)]

        elif numeral_type == self.NumeralType.Decimals:
            tokens = self.tokens(text, strategy=self.Tokenization.WhitespacePunctuation)
            return [t for t in tokens if handle_separators(t).isdecimal()]

        elif numeral_type == self.NumeralType.Digits:
            tokens = self.tokens(text, strategy=self.Tokenization.WhitespacePunctuation)
            return [t for t in tokens if handle_separators(t).isdigit()]

        elif numeral_type == self.NumeralType.Numerals:
            tokens = self.tokens(text, strategy=self.Tokenization.WhitespacePunctuation)
            return [t for t in tokens if handle_separators(t).isnumeric()]

        elif numeral_type == self.NumeralType.SpellingOutNumbers:
            return self.__extract_spelled_out_numbers(text)

        elif numeral_type == self.NumeralType.Integers:
            tokens = self.tokens(text)
            return self.__extract_integers(tokens)

        elif numeral_type == self.NumeralType.Floats:
            tokens = self.tokens(text)
            return self.__extract_floats(tokens)

        # Note, an extraction of all categories of numerals at once is not possible straight away, 
        # since there are overlaps between individual categories (e.g., Decimals_0_to_9 ⊆ Decimals ⊆ Digits ⊆ Numeric) 
        # which would then result in redundant entries. 

    def dates(self, text: str, output_format: str = "%d.%m.%Y") -> List[str]:
        """
        Extracts dates from the given text and returns a list of formatted dates.

        Parameters:
        - text (str): The input text containing dates.
        - output_format (str, optional): The desired output format for the dates. Default is "%d.%m.%Y".

        Returns:
        List[str]: A list of strings representing the extracted and formatted dates.

        Example:
        >>> text = "The event is scheduled for 01-15-2022 and 02/20/2022."
        >>> dates(text)
        ['15.01.2022', '20.02.2022']
        """
        matches = datefinder.find_dates(text)
        return [d.strftime(output_format) for d in matches]

    def urls(self, text: str) -> List[str]:
        """
        Extracts URLs from the given text.

        Parameters:
        - text (str): The input text from which URLs need to be extracted.

        Returns:
        - List[str]: A list of URLs extracted from the input text.

        Example:
        >>> text = "Visit https://www.example.com or www.beispiel.de for more information."
        >>> urls(text)
        ['https://www.example.com']
        """
        extractor = URLExtract()
        return list(extractor.gen_urls(text))

    def email_addresses(self, text: str) -> List[str]:
        """
        Extracts email addresses from the given text.

        Parameters:
        - text (str): The input text from which email addresses will be extracted.

        Returns:
        List[str]: A list of unique email addresses found in the text.

        Note:
        The function currently returns a set to ensure unique email addresses. To allow duplicates,
        modify the implementation accordingly.

        Example:
        >>> emails = email_addresses("Contact us at john.doe@example.com or support@example.com")
        ['john.doe@example.com', 'support@example.com']
        """
        # TODO: Allow duplicates !! Currently a set is returned
        return scrape_emails(text)
