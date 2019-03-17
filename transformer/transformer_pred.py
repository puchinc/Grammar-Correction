"""
Usage:
CUDA_VISIBLE_DEVICES=4 python transformer/transformer_pred.py \
    -src data/test/ \
    -model data/models/ \
    -eval data/eval/ \
    -corpus lang8_small \
    -en basic -de basic

Other options for embeddings:
    -en basic -de basic
    -en glove -de basic
    -en glove -de glove
    -en elmo -de basic
    -en elmo -de elmo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from torchtext import data, datasets
import spacy

import os
import sys
import random
import argparse

from Model import MyIterator, make_model, rebatch, batch_size_fn, greedy_decode, get_emb
from allennlp.modules.elmo import batch_to_ids

def parse_args():
    parser = argparse.ArgumentParser()        
    parser.add_argument('-src', '--SRC_DIR')
    parser.add_argument('-eval', '--EVAL_DIR')
    parser.add_argument('-model', '--MODEL_DIR')
    parser.add_argument('-corpus', '--DATA')
    parser.add_argument('-en', '--EN_EMB')
    parser.add_argument('-de', '--DE_EMB')
    args = parser.parse_args()                                      

    return args

def main():
    args = parse_args()
    SRC_DIR = args.SRC_DIR
    MODEL_DIR = args.MODEL_DIR
    EVAL_DIR = args.EVAL_DIR
    DATA = args.DATA
    EN_EMB = args.EN_EMB
    DE_EMB = args.DE_EMB

    if 'glove' in EN_EMB:
        EMB_DIM = 200
    elif 'elmo' in EN_EMB:
        EMB_DIM = 1024
    else:
        EMB_DIM = 512

    BATCH_SIZE = 30

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    vocab_file = os.path.join(MODEL_DIR, '%s.vocab' % (DATA))
    model_file = os.path.join(MODEL_DIR, '%s.%s.%s.transformer.pt' % (DATA, EN_EMB, DE_EMB))

    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)

    # GPU to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = ("cpu")

    #####################
    #   Data Loading    #
    #####################
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    MIN_FREQ = 2

    spacy_en = spacy.load('en')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    test = datasets.TranslationDataset(path=os.path.join(SRC_DIR, DATA), 
            exts=('.test.src', '.test.trg'), fields=(TEXT, TEXT))
    # use the same order as original data
    test_iter = data.Iterator(test, batch_size=BATCH_SIZE, device=device, 
                              sort=False, repeat=False, train=False)

    random_idx = random.randint(0, len(test) - 1)
    print(test[random_idx].src)
    print(test[random_idx].trg)

    ###############
    #  Vocabuary  #
    ###############
    TEXT.vocab = torch.load(vocab_file)
    pad_idx = TEXT.vocab.stoi["<blank>"]

    print("Load %s vocabuary; vocab size = %d" % (DATA, len(TEXT.vocab)))
    #####################
    #   Word Embedding  #
    #####################
    encoder_emb, decoder_emb = get_emb(EN_EMB, DE_EMB, TEXT.vocab, device, 
                                       d_model=EMB_DIM,
                                       elmo_options=options_file, 
                                       elmo_weights=weight_file)

    ##########################
    #      Translation       #
    ##########################
    model = make_model(len(TEXT.vocab), encoder_emb, decoder_emb, 
                       d_model=EMB_DIM).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    print("Predicting %s %s %s ..." % (DATA, EN_EMB, DE_EMB))

    src, trg, pred = [], [], []
    for batch in (rebatch(pad_idx, b) for b in test_iter):
        out = greedy_decode(model, TEXT.vocab, batch.src, batch.src_mask)
        # print("SRC OUT: ", src.shape, out.shape)
        probs = model.generator(out)
        _, prediction = torch.max(probs, dim = -1)

        source = [[TEXT.vocab.itos[word] for word in words[1:]] for words in batch.src]
        target = [[TEXT.vocab.itos[word] for word in words[1:]] for words in batch.trg]
        translation = [[TEXT.vocab.itos[word] for word in words] for words in prediction]

        for i in range(len(translation)):
            src.append(' '.join(source[i]).split('</s>')[0])
            trg.append(' '.join(target[i]).split('</s>')[0])
            pred.append(' '.join(translation[i]).split('</s>')[0])

            # eliminate data with unkonwn words in src trg
            if '<unk>' in src[-1] or '<unk>' in trg[-1]:
                continue

            print("Source:", src[-1])
            print("Target:", trg[-1])
            print("Translation:", pred[-1])
            print()

    prefix = os.path.join(EVAL_DIR, '%s.%s.%s.eval' % (DATA, EN_EMB, DE_EMB))
    for sentences, ext in zip([src, trg, pred], ['.src', '.trg', '.pred']):
        with open(prefix + ext, 'w+') as f:
            f.write('\n'.join(sentences))

if __name__ == "__main__":
    main()
