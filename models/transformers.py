import torch
import torch.nn as nn
import math
import numpy as np

from ..hflayers import Hopfield
from ..hflayers.transformer import HopfieldEncoderLayer
from torchtext import vocab


def load_glove(embedding_dim=300):
    print("Loading glove")
    glove_embeddings = vocab.GloVe(name='840B', dim=embedding_dim)
    print("Done loading glove")
    return nn.Embedding.from_pretrained(glove_embeddings.vectors)


class HopfieldTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=12,
        dim_feed_forward=768,
        num_layers=1,
        dropout=0.1
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList()

        hopfield_layer = Hopfield(
            input_size=d_model,
            hidden_size=dim_feed_forward,
            output_size=d_model,
            num_heads=nhead
        )

        for _ in range(num_layers):
            self.encoder_layers.append(
                HopfieldEncoderLayer(
                    hopfield_layer,
                    dim_feedforward=dim_feed_forward,
                    dropout=dropout
                )
            )

    def forward(self, src, src_mask, src_padding_mask):
        x = src
        for layer in self.encoder_layers:
            x = layer(x, src_mask, src_padding_mask)
        return x
