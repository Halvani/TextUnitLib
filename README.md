# TextUnitLib (TUL)
A Python library that allows easy extraction of a variety of text units within texts

# Description
TextUnitLib (TUL) enables effortless extraction of a variety of text units from texts, 
which can for example be used to extend existing Natural Language Processing (NLP) [applications](#Applications). In addition to common text units such as words, parts of speech (POS) or named entities, more specific text units such as function words, stop words, contractions, numerals (e.g., pronounced numbers), quotations, emojis and many more can be extracted. These can be used to carry out in-depth analyses of given texts and thus gain valuable insights (e.g., stylometric analyses). In addition, TUL can be used to simplify the pre-processing and cleaning of texts, the construction of feature vectors or many (corpus) linguistic tasks such as the creation of word/vocabulary lists, cloze texts and readability formulas. TUL can be used either standalone or as a building block for larger NLP applications. 

<a name="Applications"></a>
# Applications
TUL's feature extraction abilities allow a wide range of applications, including:

- Text analytic / corpus linguistics purposes (e.g., calculating text statistics, readability measures, authorship analysis, etc.) 
- Feature vector construction for many NLP tasks (in particular text classification)
- Accessing linguistic features for visualization purposes (e.g., word clouds, plots)
- Pre-processing / cleaning of text files within text datasets (e.g., anonymizing named entities, removing stopwords, dates, URLs, etc.)
- General framework PDF document annotations (e.g., highlighting words by their POS-tags)  

# Features
- Besides common text units (e.g., tokens, letters, numbers or POS-tags) TUL also covers many [less popular](#TextUnit_Categories) text units
- Provides functions to extract generic linguistic features (e.g., n-gram-based features, text units that occur x times, maximum substrings occuring within a list of text units, etc.)
- Multilingual (TUL currently supports two languages, more languages will follow)
- Automatic NLP pipeline creation (installation and loading of the spaCy models on demand)
- No API dependency: besides the spaCy models and obligatory Python libraries TUL can be used completely offline
- Extensively documented source code with many examples integrated into the docstrings

<a name="TextUnit_Categories"></a>
# Categories of text units
- Numerals: Integers, floats, decimals (0 to 9), digits, numerals, spelling out numbers
- Function word sub-categories: Conjunctions, auxiliary verbs, determiners, prepositions, pronouns, quantifiers 
- N-Grams: Character n-grams, word n-grams, token n-grams, POS-tag n-grams, etc.
- Emojis: As visual pictograms or as shortcodes
- Hapax/dis/tris legomenon text units 
- Quotations
