import torch
import torch.nn as nn
import math
import numpy as np

from hflayers import Hopfield
from hflayers.transformer import HopfieldEncoderLayer
from torchtext import vocab


def load_glove(embedding_dim=300):
    print("Loading glove")
    
    glove_embeddings = vocab.GloVe(name='840B', dim=embedding_dim)
    print("Done loading glove")
    return nn.Embedding.from_pretrained(glove_embeddings.vectors)


class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

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


class SSTHopfieldClassifier(nn.Module):
    def __init__(self,
        use_glove_embedding=True,
        embedding_dim=300,
        nhead=12,
        dim_feed_forward=2048,
        num_layers=12,
        dropout=0.1,
        num_classes=5,
        reduction="mean"
    ):
        super().__init__()
    
        if use_glove_embedding:
            self.embedding = load_glove(embedding_dim)
        else:
            self.embedding = nn.Embedding(400002, embedding_dim)
        
        self.positional_encoding = SinePositionalEncoding(embedding_dim, dropout=dropout)
        
        self.encoder = HopfieldTransformerEncoder(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feed_forward=dim_feed_forward,
            num_layers=num_layers,
            dropout=dropout
        )

        self.reduction = reduction
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, src, src_mask=None, src_padding_mask=None):
        x = self.embedding(src)
        x = self.encoder(x, src_mask, src_padding_mask)

        if self.reduction == "mean":
            x = x.mean(dim=0, keepdim=True).squeeze(0)
        else:
            raise ValueError(f"Invalid reduction method.")
        
        return self.fc(x)
