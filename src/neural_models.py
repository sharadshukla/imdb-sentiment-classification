"""
Neural network architectures for IMDb sentiment classification.

Models included:
- Vanilla RNN
- LSTM
- Bidirectional LSTM with optional pretrained embeddings
- Attention-based sentiment classifier
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import MAX_LEN
from .utils import positional_encoding, make_pad_mask


class RNNSentiment(nn.Module):
    """
    Vanilla RNN sentiment classifier.

    Architecture:
        Embedding
        -> 2-layer RNN
        -> final hidden state
        -> Dropout
        -> Linear classifier
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        pad_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x, lengths):
        # Convert token IDs into dense word embeddings
        embedded_tokens = self.embedding(x)

        # Ignore right-padding when processing recurrent sequences
        packed_tokens = pack_padded_sequence(
            embedded_tokens,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # h_n now corresponds to the final real token of each sequence
        _, h_n = self.rnn(packed_tokens)

        # Final hidden state from the last RNN layer
        final_hidden_state = h_n[-1]

        # Binary classification logit
        dropped_out = self.dropout(final_hidden_state)
        logits = self.fc(dropped_out)

        return logits

class LSTMSentiment(nn.Module):
    """
    LSTM sentiment classifier.

    Architecture:
        Embedding
        -> 2-layer LSTM
        -> final hidden state
        -> Dropout
        -> Linear classifier
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        pad_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=1
        )

    def forward(self, x, lengths):
        # Convert token IDs into dense word embeddings
        embedded_tokens = self.embedding(x)

        # Ignore right-padding when processing recurrent sequences
        packed_tokens = pack_padded_sequence(
            embedded_tokens,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # h_n now corresponds to the final real token of each sequence
        _, (h_n, _) = self.lstm(packed_tokens)

        # Final hidden state from the last LSTM layer
        final_hidden_state = h_n[-1]

        # Binary classification logit
        dropped_out = self.dropout(final_hidden_state)
        logits = self.fc(dropped_out)

        return logits

class BiLSTMSentiment(nn.Module):
    """
    Bidirectional LSTM sentiment classifier.

    Supports pretrained embeddings such as FastText.

    Architecture:
        Embedding
        -> Bidirectional LSTM
        -> concatenate final forward and backward states
        -> Dropout
        -> Linear
        -> ReLU
        -> Linear classifier
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        pad_idx=0,
        pretrained_embeddings=None
    ):
        super().__init__()

        # Use pretrained embeddings when supplied;
        # otherwise initialize embeddings randomly.
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,
                padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=pad_idx
            )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths):
        # Convert token IDs into embeddings
        embedded_tokens = self.embedding(x)

        # Ignore right-padding when processing recurrent sequences
        packed_tokens = pack_padded_sequence(
            embedded_tokens,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # h_n contains the final hidden states from both directions
        _, (h_n, _) = self.lstm(packed_tokens)

        # Final forward and backward states from the last BiLSTM layer
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]

        # Combine information from both directions
        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        # Binary classification logit
        output = self.classifier(combined)

        return output

class SelfAttnSentiment(nn.Module):
    """
    Attention-based sentiment classifier used in the original notebook.

    Architecture:
        Embedding + positional encoding
        -> LayerNorm + Dropout
        -> prepend learnable CLS token
        -> Multi-Head Attention
        -> CLS representation
        -> LayerNorm
        -> Linear
        -> GELU
        -> Dropout
        -> Linear classifier

    Note:
        The notebook names this model SelfAttnSentiment.
        The attention operation itself uses the same sequence for
        queries, keys, and values.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=8,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()

        # Word embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        # Fixed positional encoding
        self.register_buffer(
            "pos_enc",
            positional_encoding(
                MAX_LEN + 1,
                embedding_dim
            )
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

        # Normalization and dropout before attention
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.drop1 = nn.Dropout(dropout)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalization after attention
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # Stores attention weights for later visualization
        self.last_weights = None

    def forward(self, x, lengths=None):
        # Token embeddings
        embedded_tokens = self.embedding(x)

        # Add positional information
        embedded_tokens = (
            embedded_tokens
            + self.pos_enc[:, 1:x.shape[1] + 1, :]
        )

        embedded_tokens = self.norm1(embedded_tokens)
        embedded_tokens = self.drop1(embedded_tokens)

        # Create one CLS token for each review in the batch
        cls_token = self.cls_token.expand(
            x.shape[0],
            -1,
            -1
        )

        # Prepend CLS token to the token sequence
        seq = torch.cat(
            [cls_token, embedded_tokens],
            dim=1
        )

        # Attention over the complete review
        out, weights = self.attention(
            seq,
            seq,
            seq,
            key_padding_mask=make_pad_mask(x),
            need_weights=True,
            average_attn_weights=True
        )

        # Save attention weights for visualization
        self.last_weights = weights.detach().cpu()

        # CLS representation summarizes the review
        cls_repr = self.norm2(
            out[:, 0, :]
        )

        # Binary classification logit
        return self.classifier(cls_repr)