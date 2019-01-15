import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DATA ###
# Small dataset
# small = True
# path = 'CoNLL_data/train.txt'
# emb_path = 'CoNLL_data/train_small.elmo'
# emb_path = 'CoNLL_data/train_small.elmo.pair'

# Baseline
# small = False
# path = 'CoNLL_data/train_baseline.txt'
# emb_path = 'CoNLL_data/train_baseline.elmo'

# Add error tag
# path = 'CoNLL_data/train.txt'
# emb_path = 'CoNLL_data/train.elmo'

### MODEL ###

# dir_path = os.path.dirname(os.path.realpath(__file__))
# encoder_path = os.path.join(dir_path, 'models/with_error_tag.encoder')
# decoder_path = os.path.join(dir_path, 'models/with_error_tag.decoder')

hidden_size = 256
elmo_size = 1024

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100

### TRAIN ###
teacher_forcing_ratio = 0.5

print_every = 100
n_iters = 2000
