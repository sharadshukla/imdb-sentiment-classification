"""
Data loading and PyTorch dataset utilities for IMDb sentiment classification.
"""

import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from .preprocessing import tokenize


# Sequence and batch settings used in the original experiment
MAX_LEN = 400
BATCH_SIZE = 64


def load_imdb_data():
    """
    Load the Stanford IMDb dataset and return the train/test texts and labels.

    Returns
    -------
    train_texts : list[str]
        Training review texts.
    y_train : np.ndarray
        Training labels.
    test_texts : list[str]
        Test review texts.
    y_test : np.ndarray
        Test labels.
    """
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    return train_texts, y_train, test_texts, y_test


class IMDBDataset(Dataset):
    """
    Convert IMDb reviews and labels into fixed-length PyTorch tensors.

    Reviews longer than max_len are truncated.
    Shorter reviews are padded with the PAD token (ID 0).
    Unknown words are mapped to the UNK token (ID 1).

    The true sequence length is preserved so recurrent models can ignore
    padded positions during training and evaluation.
    """

    def __init__(self, texts, labels, word2idx, max_len=MAX_LEN):
        self.labels = labels
        self.seqs = []
        self.lengths = []

        for doc in texts:
            ids = [
                word2idx.get(token, 1)
                for token in tokenize(doc)
            ]

            # Truncate long reviews
            ids = ids[:max_len]

            # Avoid zero-length sequences for recurrent models
            if not ids:
                ids = [1]

            # Preserve the true sequence length before padding
            length = len(ids)

            # Pad short reviews
            ids += [0] * (max_len - length)

            self.seqs.append(ids)
            self.lengths.append(length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seqs[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

def create_dataloaders(
    train_texts,
    y_train,
    test_texts,
    y_test,
    word2idx,
    max_len=MAX_LEN,
    batch_size=BATCH_SIZE
):
    """
    Create the PyTorch datasets and dataloaders used by the neural models.
    """

    train_ds = IMDBDataset(
        train_texts,
        y_train,
        word2idx,
        max_len
    )

    test_ds = IMDBDataset(
        test_texts,
        y_test,
        word2idx,
        max_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, test_ds, train_loader, test_loader