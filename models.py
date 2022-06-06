import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from hflayers import Hopfield
from hflayers.transformer import HopfieldEncoderLayer


def load_glove(embedding_dim=300, checkpoint_path=".vector_cache"):
    print("Loading glove")

    vocab, embeddings = [], []
    with open(f"{checkpoint_path}/glove.6B.{embedding_dim}d.txt", "rt") as f:
        full_content = f.read().strip().split("\n")

    for i in range(len(full_content)):
        i_word = full_content[i].split(" ")[0]
        i_embeddings = [float(val) for val in full_content[i].split(" ")[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    vocab_npa = np.insert(vocab_npa, 0, "<pad>")
    vocab_npa = np.insert(vocab_npa, 1, "<cls>")
    vocab_npa = np.insert(vocab_npa, 2, "<unk>")

    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)
    cls_emb_npa = np.ones((2, embs_npa.shape[1]))

    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, cls_emb_npa, embs_npa))

    print("Done loading glove")

    return nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())


class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class HopfieldTransformerEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=12, dim_feed_forward=2048, num_layers=12, dropout=0.1
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList()

        hopfield_layer = Hopfield(
            input_size=d_model,
            hidden_size=dim_feed_forward,
            output_size=d_model,
            num_heads=nhead,
        )

        for _ in range(num_layers):
            self.encoder_layers.append(
                HopfieldEncoderLayer(
                    hopfield_layer, dim_feedforward=dim_feed_forward, dropout=dropout
                )
            )

    def forward(self, src, src_mask, src_padding_mask):
        x = src

        for layer in self.encoder_layers:
            x = layer(x, src_mask, src_padding_mask)

        return x


class SSTHopfieldClassifier(nn.Module):
    def __init__(
        self,
        use_glove_embedding=False,
        embedding_dim=300,
        nhead=12,
        dim_feed_forward=2048,
        num_layers=12,
        dropout=0.1,
        num_classes=5,
        reduction="mean",
    ):
        super().__init__()

        if use_glove_embedding:
            self.embedding = load_glove(embedding_dim)
        else:
            self.embedding = nn.Embedding(250883, embedding_dim)

        self.positional_encoding = SinePositionalEncoding(
            embedding_dim, dropout=dropout
        )

        self.encoder = HopfieldTransformerEncoder(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feed_forward=dim_feed_forward,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.reduction = reduction
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, src, src_mask=None, src_padding_mask=None):
        x = self.embedding(src)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_mask, src_padding_mask)

        if self.reduction == "mean":
            x = x.mean(dim=0, keepdim=True).squeeze(0)
        elif self.reduction == "first":
            x = x[0, :, :].squeeze(0)
        elif self.reduction != "none":
            raise ValueError(f"Invalid reduction method.")

        x = self.fc(F.relu(x))

        return x


class BERTClassifier(nn.Module):
    def __init__(
        self,
        use_glove_embedding=False,
        embedding_dim=300,
        nhead=12,
        dim_feed_forward=2048,
        num_layers=12,
        dropout=0.1,
        num_classes=5,
        reduction="mean",
    ):
        super().__init__()

        if use_glove_embedding:
            self.embedding = load_glove(embedding_dim)
        else:
            self.embedding = nn.Embedding(250883, embedding_dim)

        self.positional_encoding = SinePositionalEncoding(
            embedding_dim, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feed_forward,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reduction = reduction
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, src, src_mask=None, src_padding_mask=None):
        x = self.embedding(src)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_mask, src_padding_mask)

        if self.reduction == "mean":
            x = x.mean(dim=0, keepdim=True).squeeze(0)
        elif self.reduction == "first":
            x = x[0, :, :].squeeze(0)
        elif self.reduction != "none":
            raise ValueError(f"Invalid reduction method.")

        x = self.fc(F.relu(x))

        return x
