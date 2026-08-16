"""
Text preprocessing utilities for IMDb sentiment classification.

This module contains:
- basic text cleaning used by the classical ML models
- tokenization used by the neural models
- vocabulary construction for mapping tokens to integer IDs
"""

import re
from collections import Counter


# Minimum number of occurrences required for a word
# to be included in the neural-model vocabulary.
MIN_FREQ = 5


def clean_text(text):
    """
    Clean text for the classical machine-learning pipeline.

    The cleaning follows the original notebook:
    - remove HTML tags
    - convert text to lowercase
    - strip leading/trailing whitespace
    """

    # Remove HTML tags
    cleaned_text = re.sub(r"<.*?>", " ", text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def tokenize(text):
    """
    Tokenize text for the neural models.

    The tokenizer:
    - removes HTML tags
    - converts text to lowercase
    - removes punctuation
    - splits the text into individual tokens
    """

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and keep letters, numbers, and whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Split into tokens
    return text.split()


def build_vocabulary(texts, min_freq=MIN_FREQ):
    """
    Build a vocabulary from training texts only.

    Words appearing fewer than min_freq times are mapped to <UNK>.

    Returns
    -------
    word2idx : dict
        Maps words to integer token IDs.

    idx2word : dict
        Maps integer token IDs back to words.
    """

    counter = Counter()

    for doc in texts:
        counter.update(tokenize(doc))

    # Reserve:
    # 0 -> padding token
    # 1 -> unknown token
    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    idx2word = {
        idx: word
        for word, idx in word2idx.items()
    }

    return word2idx, idx2word