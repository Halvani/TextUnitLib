# TextUnitLib (TUL)
A Python library that allows easy extraction of a variety of text units within texts

## Description
TextUnitLib (**TUL**) provides an easy and flexible way to extract a wide range of text units from raw text. These units can be used to enhance existing Natural Language Processing (NLP) [applications](#Applications) or to support standalone analyses. In addition to common units such as words, parts of speech (POS), or named entities, TUL also handles more specialized categories including function words, stopwords, contractions, numerals (e.g., spoken numbers), quotations, emojis, and many others.

These extracted units enable in-depth text analyses and support tasks such as stylometric investigations, corpus-linguistic studies, and data exploration. TUL also simplifies text pre-processing and cleaning, facilitates feature-vector construction for NLP models, and supports various corpus-oriented workflows such as generating vocabulary lists, cloze texts, or readability metrics.

TUL can be used as a standalone toolkit or integrated as a component within larger NLP systems.

## Installation
The easiest way to install TextUnitLib is to use pip, where you can choose between (1) the PyPI repository and (2) this repository. 

- (1) ```pip install textunitlib```

- (2) ```pip install git+https://github.com/Halvani/TextUnitLib.git ```

The latter will pull and install the latest commit from this repository as well as the required Python dependencies.

## Quickstart
Below you can find several quick examples how to use TUL to extract various text units from given texts.


### Initialize a TextUnit instance
```python
from textunitlib import TextUnit

# Creates an NLP pipeline with a small English spaCy model by default.
tu = TextUnit()

# To load a specific spaCy model:
tu = TextUnit(model_id=TextUnit.SpacyModelSize.English_Large)
```

### Extract function words
```python
text =  "The kickoff meeting will take place on Tuesday."

print(tu.function_words(text))

# ['The', 'will', 'on']
```

### Extract contractions
```python
text_contractions = """I’m pretty sure we’ll finish this today, but if we don’t, that’s alright — we’ve still got tomorrow. You shouldn’t worry too much; it isn’t as hard as it looks, and they’ve already done most of the work anyway."""

print(tu.contractions(text_contractions, full_form=True))

# ['I am', 'we will', 'that is', 'we have', 'should not', 'is not', 'they have']
```

### Extracting social media hashtags and emojis
```python
text_social_media = """Just finished my #morningrun and feeling amazing! 🌞💪 
Time to grab some #coffee and tackle the #MondayMotivation vibes. 
Who else is ready to #crushit today? #StayPositive #Goals""" 

print(tu.emojis(text_social_media))
# ['🌞', '💪']

print(tu.hashtags(text_social_media))
# ['#morningrun', '#coffee', '#MondayMotivation', '#crushit', '#StayPositive', '#Goals']
```

### Extract named entities (NEs)
```python
text_named_entities = """In April 2024, Dr. Maria Sánchez from Stanford University met with engineers at OpenAI in San Francisco to discuss a collaboration on Project Helios. Later that year, Tesla Inc. announced plans to integrate their findings into autonomous vehicles across North America."""

# Extract all NEs

print(tu.named_entities(text_named_entities))
# ['April 2024', 'Maria Sánchez', 'Stanford University', 'OpenAI', 'San Francisco', 'Project Helios', 'Later that year', 'Tesla Inc.', 'North America']

# Restrict the extracted NEs to a given list of categories. 
# Available labels shown below.

tu.named_entities(text_named_entities, restrict_to_categories=["ORG"])
# ['Stanford University', 'Project Helios', 'Tesla Inc.']
```

<details>
<summary><strong>Show Available NEs</strong></summary>

<br>

| Label           | Description                                                |
|---------------- | ---------------------------------------------------------- |
| **CARDINAL**    | Numerals that do not fall under another type               |
| **DATE**        | Absolute or relative dates and periods                     |
| **EVENT**       | Named events (wars, sports events, disasters, etc.)        |
| **FAC**         | Facilities such as buildings, airports, highways, bridges  |
| **GPE**         | Countries, cities, states (geopolitical entities)          |
| **LANGUAGE**    | Named languages                                            |
| **LAW**         | Named legal documents or laws                              |
| **LOC**         | Non-GPE locations (mountain ranges, bodies of water, etc.) |
| **MONEY**       | Monetary values, including currency units                  |
| **NORP**        | Nationalities, religious or political groups               |
| **ORDINAL**     | “first”, “second”, etc.                                    |
| **ORG**         | Organizations (companies, institutions, agencies)          |
| **PERCENT**     | Percentage values, including the “%” symbol                |
| **PERSON**      | People, including fictional characters                     |
| **PRODUCT**     | Physical or digital products                               |
| **QUANTITY**    | Measurements (weight, distance, etc.)                      |
| **TIME**        | Times smaller than a day                                   |
| **WORK_OF_ART** | Titles of books, songs, movies, etc.                       |

</details>


### Extract dates
```python
text_dates = """The first prototype was released on 2021-07-15, and version 1.0 followed on July 20, 2022. A major update arrived on 15/08/2023, just before the annual review on 08.09.2023. Our next release is scheduled for March 1st, 2024, with a beta planned for 01 March 2024. Please submit your reports by 12/31/2024 or, at the latest, by 2025/01/10. The kickoff meeting took place on Tuesday, 3 January 2023, and follow-ups are held every Monday."""

# Preserve original format of the extracted dates (not bullet-proof)

print(tu.dates(text_dates, preserve_input_format=True))
# ['on 2021-07-15', 'on 1.0', 'wed on July 20, 2022', 'on 15/08/2023', 'on 08.09.2023', 'March 1st, 2024', '01 March 2024', 'by 12/31/2024', 'by 2025/01/10', 'on Tuesday, 3 January 2023', 'Monday']

# Unify extracted dates to the format "dd.mm.yyyy" (default)

print(tu.dates(text_dates))
# ['15.07.2021', '01.11.2025', '20.07.2022', '15.08.2023', '09.08.2023', '01.03.2024', '01.03.2024', '31.12.2024', '10.01.2025', '03.01.2023', '17.11.2025']
```

### Extract stop words (superset of function words)
```python
text =  "The kickoff meeting will take place on Tuesday."

print(tu.stop_words(text))
# ['The', 'will', 'take', 'on']
```

### Extract Part of Speech tags (POS tags)
```python
text =  "The kickoff meeting will take place on Tuesday."

# Extract all POS tags

print(tu.postags(text))
# ['DET', 'NOUN', 'NOUN', 'AUX', 'VERB', 'NOUN', 'ADP', 'PROPN', 'PUNCT']

# Extract all POS tags and combine them with corresponding tokens

print(tu.postags(text, combine_with_token=True, combine_sep=" "))
# [('The', 'DET'), ('kickoff', 'NOUN'), ('meeting', 'NOUN'), ('will', 'AUX'), ('take', 'VERB'), ('place', 'NOUN'), ('on', 'ADP'), ('Tuesday', 'PROPN'), ('.', 'PUNCT')]

# Extract only nouns and return their tokens instead of the POS tags

print(tu.postags(text, tokens_only=True, tags_to_consider={"NOUN"}))
# ['kickoff', 'meeting', 'place']
```


<a name="Applications"></a>
## Applications
TUL’s feature extraction capabilities enable a wide range of use cases, including:

* **Text analytics and corpus linguistics**, such as computing text statistics, readability scores, or authorship-related features
* **Feature vector construction** for a variety of NLP tasks, particularly text classification
* **Linguistic feature access for visualization**, including word clouds, plots, and exploratory data analysis
* **Text pre-processing and cleaning** in larger datasets, such as anonymizing named entities, removing stopwords, dates, or URLs
* **PDF document annotation workflows**, such as highlighting tokens according to their POS tags

## Features
* **Wide coverage:** In addition to common text units (e.g., tokens, letters, numbers, POS tags), TUL also supports many [less commonly used](#TextUnit_Categories) unit types.
* **Practical helper functions:** Includes utilities for extracting general linguistic features such as n-grams, frequency-based units, and maximal substrings across text-unit sequences.
* **Multilingual:** Currently supports two languages, with additional languages planned.
* **Automatic NLP pipeline setup:** Automatically loads and installs required spaCy models on demand and provides progress feedback.
* **Offline-capable:** Aside from spaCy models and standard Python libraries, TUL has no external API dependencies and works fully offline.
* **Documentation with examples:** Thoroughly documented source code, enriched with numerous usage examples embedded in docstrings.

<a name="TextUnit_Categories"></a>
## Categories of text units
- Numerals: Integers, floats, decimals (0 to 9), digits, numerals, spelling out numbers
- Function word sub-categories (conjunctions, auxiliary verbs, determiners, prepositions, pronouns, quantifiers) 
- N-Grams: Character n-grams, word n-grams, token n-grams, POS-tag n-grams, etc.
- Emojis: As visual pictograms or as shortcodes
- Hapax/dis/tris legomenon text units 
- Quotations
