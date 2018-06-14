# Need allennlp with torch==0.3.1 && GPU

import pickle
import re
import unicodedata

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^\-</>a-zA-Z.!?]+", r" ", s)
    return s

def loadConll(path='CoNLL_data/train.txt'):
    # Read the file and split into lines
    path='CoNLL_data/train.txt'
    lines = open(path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize and then split to words
    pairs = [[[e for e in normalizeString(s).split(' ')] for s in l.split('\t')] for l in lines]
    return pairs

def elmoFromPair(pair):
    # choose layer 2 representations
    layer = 1
    character_ids = batch_to_ids(pair)
    input_tensor, output_tensor = elmo(character_ids)['elmo_representations'][layer].data
    return (input_tensor, output_tensor)

def sen2elmo(sentence):
    layer = 1
    character_ids = batch_to_ids(sentence)
    tensor = elmo(character_ids)['elmo_representations'][layer][0].data
    return tensor

# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
if __name__ == '__main__':
    # TEST SMALL DATASET
    # path = 'CoNLL_data/train.txt'
    # emb_path = 'CoNLL_data/train_small.elmo'
    # pairs = loadConll(path)[:10]

    # BASELINE
    # path = 'CoNLL_data/baseline_train.txt'
    # emb_path = 'CoNLL_data/train_baseline.elmo'

    # WITH ERROR TAG
    path = 'CoNLL_data/train.txt'
    emb_path = 'CoNLL_data/train.elmo'

    # pairs = [[['First', 'sentence', '.'], ['Another', '.']]]
    pairs = loadConll(path)

    # (elmo, text) pairs
    embeddings = [[sen2elmo(pair[0]), pair[1]] for pair in pairs]
    with open(emb_path, 'wb') as file:
        pickle.dump(embeddings, file)
