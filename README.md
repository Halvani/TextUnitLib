# TextUnitLib (TUL)
A Python library that allows convenient access to a variety of text units from given texts.

# Description
TUL provides a convenient way to extract various text units from texts and is particularly useful for text analytic purposes.

# Applications
TUL's feature extraction abilities allow a wide range of applications, including:
- Feature vector construction for many NLP tasks (in particular text classification)
- Calculating text statistics for various linguistic use cases
- Accessing linguistic features for visualization purposes (e.g., word clouds, plots) or for annotations within PDF documents
- Search and replace of categorized text units in text documents (e.g., masking all verbs in a )

# Features
- Besides common text units (e.g., tokens, letters, numbers, digits or part-of-speech tags) TUL also covers many less popular text units (see here)
- Allows the extraction of generic linguistic features (e.g. n-grams, text units that occur x times, etc.)
- Multilingual (TUL currently supports two languages, more languages will follow)
- Automatic NLP pipeline creation (loads and installs the spaCy models on demand)
- No API dependency: besides the spaCy models and the obligatory dependencies (in the form of Python libraries TUL can be used completely offline
- Extensively documented source code with many examples integrated into the docstrings


