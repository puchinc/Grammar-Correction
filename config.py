import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOS_token = torch.zeros([1, 1024], device=device)
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100
