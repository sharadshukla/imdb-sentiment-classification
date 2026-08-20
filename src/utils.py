"""
Utility functions for IMDb sentiment classification.

Includes:
- neural training-loss visualization
- trainable parameter counting
- FastText embedding loading and matrix construction
- positional encoding
- padding-mask creation for self-attention
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch


# FastText embedding dimension used in the original experiment
FASTTEXT_EMBED_DIM = 300


def plot_losses(losses, title, output_path=None):
    """
    Plot the training loss across epochs.

    If output_path is supplied, the figure is saved to disk.
    """

    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, losses, marker="o")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Loss curve saved to: {output_path}")

    plt.close()


def count_params(model):
    """
    Count and print the trainable parameters in a PyTorch model.
    """

    n = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"  Trainable parameters: {n:,}")

    return n


def load_fasttext_embeddings(
    model_name="fasttext-wiki-news-subwords-300"
):
    """
    Load the pretrained FastText word vectors used in the experiment.

    The model is downloaded by gensim the first time it is requested
    and then loaded from the local cache on subsequent runs.
    """

    import gensim.downloader as api

    print(
        "Loading FastText embeddings "
        "(downloads once, ~1 GB, then cached)..."
    )

    ft_model = api.load(model_name)

    print(
        f"FastText loaded. Vocabulary: {len(ft_model):,} words, "
        f"dim: {FASTTEXT_EMBED_DIM}"
    )

    return ft_model


def build_embedding_matrix(
    word2idx,
    ft_model,
    dim=FASTTEXT_EMBED_DIM
):
    """
    Build an embedding matrix aligned with the project vocabulary.

    Words available in FastText receive their pretrained vectors.
    Words without a FastText vector remain randomly initialized.
    The PAD token (index 0) is always kept as a zero vector.
    """

    matrix = np.random.uniform(
        -0.1,
        0.1,
        (len(word2idx), dim)
    ).astype(np.float32)

    found = 0

    for word, idx in word2idx.items():
        # PAD must remain a zero vector
        if idx == 0:
            continue

        if word in ft_model:
            matrix[idx] = ft_model[word]
            found += 1

    # Explicitly enforce the PAD invariant after matrix construction
    matrix[0] = 0.0

    coverage = found / len(word2idx) * 100

    print(
        f"Coverage: {found}/{len(word2idx)} vocab words "
        f"have a FastText vector ({coverage:.1f}%)"
    )

    return torch.FloatTensor(matrix)

def positional_encoding(max_len, d_model):
    """
    Create sinusoidal positional encodings.

    Self-attention has no built-in knowledge of token order,
    so this encoding provides a unique position-dependent signal
    for each token.
    """

    pe = torch.zeros(max_len, d_model)

    pos = torch.arange(
        0,
        max_len,
        dtype=torch.float
    ).unsqueeze(1)

    div = torch.exp(
        torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float
        )
        * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(pos * div)

    pe[:, 1::2] = torch.cos(
        pos * div[:d_model // 2]
    )

    return pe.unsqueeze(0)

def make_pad_mask(x):
    """
    Create the padding mask used by the self-attention model.

    True indicates a PAD token that attention should ignore.
    A False column is prepended because the CLS token is never padding.
    """

    batch_size = x.shape[0]

    # True wherever token ID = 0 (<PAD>)
    pad = x == 0

    # CLS token is never padding
    cls = torch.zeros(
        batch_size,
        1,
        dtype=torch.bool,
        device=x.device
    )

    return torch.cat(
        [cls, pad],
        dim=1
    )