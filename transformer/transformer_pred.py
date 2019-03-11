# Reference: 
# codebase: http://nlp.seas.harvard.edu/2018/04/03/attention.html
# torchtext load pretrained embeddings: http://anie.me/On-Torchtext/

# Prelims:
# pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
# python -m spacy download en 

# Train:
# python transformer_pred.py

# Evaluate:
# python ../evaluation/gleu.py -s source.txt -r target.txt --hyp pred.txt

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
from pprint import pprint

from Model import MyIterator, make_model, rebatch, batch_size_fn, greedy_decode, get_emb
from allennlp.modules.elmo import batch_to_ids

def main():
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    DATA = 'lang8_small'
    # EMB_DIM should be multiple of 8, look at MultiHeadedAttention
    # EN_EMB, DE_EMB, EMB_DIM = 'basic', 'basic', 512
    # EN_EMB, DE_EMB, EMB_DIM = 'glove', 'basic', 200
    # EN_EMB, DE_EMB, EMB_DIM = 'glove', 'glove', 200
    EN_EMB, DE_EMB, EMB_DIM = 'elmo', 'basic', 1024
    # EN_EMB, DE_EMB, EMB_DIM = 'elmo', 'elmo', 1024

    BATCH_SIZE = 30

    # GPU to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = ("cpu")

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    src_dir = os.path.join(root_dir, 'data/src')
    test_dir = os.path.join(root_dir, 'data/test')
    eval_dir = os.path.join(root_dir, 'data/eval')
    vocab_file = os.path.join(root_dir, 'data/models', '%s.vocab' % (DATA))

    elmo_options_file = os.path.join(root_dir, 'data/embs/elmo.json')
    elmo_weights_file = os.path.join(root_dir, 'data/embs/elmo.hdf5')
    model_file = os.path.join(root_dir, 'data/models', '%s.%s.%s.transformer.pt' % (DATA, EN_EMB, DE_EMB))

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    #####################
    #   Data Loading    #
    #####################
    spacy_en = spacy.load('en')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    test = datasets.TranslationDataset(path=os.path.join(src_dir, DATA), 
            exts=('.test.src', '.test.trg'), fields=(TEXT, TEXT))
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
                                       elmo_options=elmo_options_file, 
                                       elmo_weights=elmo_weights_file)

    ##########################
    #      Translation       #
    ##########################
    model = make_model(len(TEXT.vocab), encoder_emb, decoder_emb, 
                       d_model=EMB_DIM).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    f_src = open(os.path.join(eval_dir, 
        '%s.%s.%s.eval.src' % (DATA, EN_EMB, DE_EMB)), 'w+')
    f_trg = open(os.path.join(eval_dir, 
        '%s.%s.%s.eval.trg' % (DATA, EN_EMB, DE_EMB)), 'w+')
    f_pred = open(os.path.join(eval_dir, 
        '%s.%s.%s.eval.pred' % (DATA, EN_EMB, DE_EMB)), 'w+')

    print("Predicting %s %s %s ..." % (DATA, EN_EMB, DE_EMB))
    for batch in (rebatch(pad_idx, b) for b in test_iter):
        out = greedy_decode(model, TEXT.vocab, batch.src, batch.src_mask)
        # print("SRC OUT: ", src.shape, out.shape)
        probs = model.generator(out)
        _, pred = torch.max(probs, dim = -1)

        source = [[TEXT.vocab.itos[word] for word in words[1:]] for words in batch.src]
        target = [[TEXT.vocab.itos[word] for word in words[1:]] for words in batch.trg]
        translation = [[TEXT.vocab.itos[word] for word in words] for words in pred]

        for i in range(len(translation)):
            src = ' '.join(source[i]).split('</s>')[0]
            trg = ' '.join(target[i]).split('</s>')[0]
            pred = ' '.join(translation[i]).split('</s>')[0]

            if '<unk>' in src or '<unk>' in trg:
                continue

            print("Source:", src)
            print("Target:", trg)
            print("Translation:", pred)
            print()

            f_src.write(src + '\n')
            f_trg.write(trg + '\n')
            f_pred.write(pred + '\n')

    f_src.close()
    f_trg.close()
    f_pred.close()

if __name__ == "__main__":
    main()
