# textunitlib/lif_functions.py

from enum import Enum
from typing import Union, List


class ResultingGrams(Enum):
    TOKENS = "tokens"
    CHARACTERS = "characters"


def textunit_ngrams(
    text_units: Union[str, List[str]],
    n: int,
    sep: str = " ",
    strip_spaces: bool = False,
    resulting_grams: ResultingGrams = ResultingGrams.CHARACTERS
) -> List[str]:
    """
    Generate n-grams from a given input, either characterwise or tokenwise.

    Parameters:
        text_units (Union[str, List[str]]): Input text units. Can be a string or list of tokens.
        n (int): Size of the n-grams to generate.
        sep (str, optional): Separator between elements in an n-gram (used only for characterwise mode).
            Default is a space. If not provided explicitly and resulting_grams=CHARACTERS, it defaults to "".
        strip_spaces (bool, optional): If True, strip leading and trailing spaces from each n-gram. Default is False.
        resulting_grams (ResultingGrams, optional): Determines whether to generate n-grams characterwise or tokenwise.
            - ResultingGrams.CHARACTERS: treat input as a sequence of characters.
            - ResultingGrams.TOKENS: treat input as a sequence of tokens (words).

    Returns:
        List[str]: A list of n-grams generated from the input text.

    Raises:
        ValueError: If n < 1 or larger than the number of units in the input.

    Examples:
        >>> textunit_ngrams("here you go", n=2, resulting_grams=ResultingGrams.TOKENS)
        ['here you', 'you go']

        >>> textunit_ngrams("here you go", n=2, resulting_grams=ResultingGrams.CHARACTERS, sep=" ")
        ['here you', 'you go']
    """

    if resulting_grams == ResultingGrams.TOKENS:
        # Token-based mode: split string into tokens if needed
        if isinstance(text_units, str):
            text_units = text_units.split()
        # Ignore sep in token mode (always join with a single space)
        sep = " "

    elif resulting_grams == ResultingGrams.CHARACTERS:
        # Character-based mode
        if isinstance(text_units, str):
            text_units = list(text_units)
        # If the caller used the default sep (space), set it to empty string
        if sep == " ":
            sep = ""

    length = len(text_units)
    if n < 1:
        raise ValueError("Note, n must be at least 1 (n >= 1).")
    if n > length:
        raise ValueError("Cannot generate n-grams larger than the number of text units.")

    ngrams = [sep.join(text_units[i:i + n]) for i in range(length - n + 1)]

    if strip_spaces:
        ngrams = [ng.strip() for ng in ngrams]

    return ngrams