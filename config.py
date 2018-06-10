import torch
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )
