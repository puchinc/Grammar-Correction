import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
elmo_size = 1024

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100

### TRAIN ###
teacher_forcing_ratio = 0.5

print_every = 100
n_iters = 2000
