import torch
from torch import nn as nn

from .highway_network import HighwayNetwork
from .initialized_conv import Initialized_Conv1d, Initialized_Conv2d


class Embedding(nn.Module):
    def __init__(self,
                 word_vectors, char_vectors,
                 model_dim,
                 word_dropout=0.1, char_dropout=0.05, freeze_char_embedding=False):
        super().__init__()

        self.word_embedding = WordEmbedding(word_vectors, model_dim, word_dropout)
        self.char_embedding = CharEmbedding(char_vectors, model_dim, char_dropout, freeze_char_embedding)
        self.highway = HighwayNetwork(dropout=word_dropout, hidden_size=2 * model_dim, num_layers=2)
        self.embedding_conv = Initialized_Conv1d(2 * model_dim, model_dim, bias=False)


    def forward(self, w_ids, c_ids):
        char_embedding = self.char_embedding(c_ids)
        word_embedding = self.word_embedding(w_ids)

        embedding = torch.cat([char_embedding, word_embedding], dim=1)
        embedding = self.highway(embedding)
        embedding = self.embedding_conv(embedding)

        return embedding

class WordEmbedding(nn.Module):
    def __init__(self,
                 word_vectors, model_dim, dropout):
        super().__init__()

        self.lookup = nn.Embedding.from_pretrained(word_vectors)

        embed_dim = word_vectors.shape[-1]
        self.conv = Initialized_Conv1d(embed_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        embedding = self.lookup(token_ids)
        embedding = self.dropout(embedding)
        embedding = self.conv(embedding.permute(0, 2, 1))

        return embedding

class CharEmbedding(nn.Module):
    def __init__(self,
                 char_vectors, model_dim, dropout,
                 freeze_char_embedding=False):
        super().__init__()

        self.lookup = nn.Embedding.from_pretrained(char_vectors, freeze=freeze_char_embedding)

        embed_dim = char_vectors.shape[-1]
        self.conv = Initialized_Conv2d(embed_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        embedding = self.lookup(char_ids)
        embedding = self.dropout(embedding)
        embedding = self.conv(embedding.permute(0, 3, 1, 2))
        embedding, _ = torch.max(embedding, dim=-1)

        return embedding
