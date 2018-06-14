from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import *

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embeddings is a torch tensor.
        # embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
        # embedding.weight = nn.Parameter(embeddings)

        # self.embedding = nn.Embedding(input_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)

        # Elmo embedding
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        # input of shape (seq_len, batch, input_size)
        # embedded = self.embedding(input).view(1, 1, -1)

        # USE ELMO
        embedded = input.view(1, 1, -1)

        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
