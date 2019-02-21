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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_type='default'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding_type = embedding_type
        
        if self.embedding_type == 'nn.embedding':
            # use nn.embedding
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
        else:
            # use ELMo or bert embedding
            self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        if self.embedding_type == 'nn.embedding':
            # use nn.embedding
            embedded = self.embedding(input).view(1, 1, -1)
        else:
            # use ELMo or bert embedding
            embedded = input.view(1, 1, -1)

        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
