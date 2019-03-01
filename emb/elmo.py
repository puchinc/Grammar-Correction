# Need allennlp with torch==0.3.1 && GPU

# Reference:
# https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md

# Example Usage:
#   python emb/elmo.py data/test/conll.txt data/embs/conll.elmo


import os
import sys
import pickle
import re
import unicodedata

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

def loadConll(path):
    # Read the file and split into lines
    lines = open(path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize and then split to words
    pairs = [[[e for e in normalizeString(s).split(' ')] for s in l.split('\t')] for l in lines]
    return pairs

def elmoFromPair(pair, elmo):
    # choose layer 2 representations
    layer = 1
    character_ids = batch_to_ids(pair)
    input_tensor, output_tensor = elmo(character_ids)['elmo_representations'][layer].data
    return (input_tensor, output_tensor)

def sen2elmo(sentence, elmo):
    layer = 1
    character_ids = batch_to_ids(sentence)
    tensor = elmo(character_ids)['elmo_representations'][layer][0].data
    return tensor

# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        sentence_path = sys.argv[1]
        emb_path = sys.argv[2]
    else:
        # TEST SMALL DATASET
        sentence_path = '../data/src/conll.txt'
        emb_path = '../data/embs/conll.elmo'
        pairs = loadConll(sentence_path)[:10]

    # no need to pretrain again
    if os.path.isfile(emb_path): 
        exit()

    emb_dir = os.path.dirname(os.path.abspath(emb_path))
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    
    from allennlp.modules.elmo import Elmo, batch_to_ids
    options_file = "../data/embs/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
    weight_file = "../data/embs/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    # explaination about num_output_representation = 2
    # https://github.com/allenai/allennlp/issues/1516
    elmo = Elmo(options_file, weight_file, 2, dropout=0)

    # BASELINE
    # sentence_path = 'CoNLL_data/baseline_train.txt'
    # emb_path = 'CoNLL_data/train_baseline.elmo'

    # WITH ERROR TAG
    # sentence_path = 'CoNLL_data/train.txt'
    # emb_path = 'CoNLL_data/train.elmo'

    # pairs = [[['First', 'sentence', '.'], ['Another', '.']]]
    pairs = loadConll(sentence_path)[:100]

    # (elmo, text) pairs
    embeddings = [[sen2elmo(pair[0], elmo), pair[1]] for pair in pairs]
    # (elmo, elmo) pairs
    # embeddings = [elmoFromPair(pair, elmo) for pair in pairs]

    with open(emb_path, 'wb') as file:
        pickle.dump(embeddings, file)
