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
    # if 'glove' in [EN_EMB, DE_EMB]:
        # vocab_file = os.path.join(root_dir, 'data/models', '%s.glove.vocab' % (DATA))

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
    # if 'glove' in [EN_EMB, DE_EMB]:
        # TEXT.build_vocab(test.src, vectors='glove.6B.200d')
    # else:
        # TEXT.vocab = torch.load(vocab_file)
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

    print("Predicting %s %s %s ..." % (DATA, EN_EMB, DE_EMB))

    for batch in (rebatch(pad_idx, b) for b in test_iter):
        out = greedy_decode(model, TEXT.vocab, batch.src, batch.src_mask)
        # print("SRC OUT: ", src.shape, out.shape)
        probs = model.generator(out)
        _, pred = torch.max(probs, dim = -1)

        trans = [[TEXT.vocab.itos[word] for word in words] for words in pred]
        source = [[TEXT.vocab.itos[word] for word in words] for words in batch.src]
        target = [[TEXT.vocab.itos[word] for word in words] for words in batch.trg]
        print("Source:", ' '.join(source[0]).split('</s>')[0])
        print("Target:", ' '.join(target[0]).split('</s>')[0])
        print("Translation:", ' '.join(trans[0]).split('</s>')[0])
        print()

    sys.exit()

    # f_src = open(os.path.join(eval_dir, 'lang8.eval.src'), 'w+')
    # f_trg = open(os.path.join(eval_dir, 'lang8.eval.trg'), 'w+')
    # f_pred = open(os.path.join(eval_dir, 'lang8.eval.pred'), 'w+')

    # for i, batch in enumerate(test_iter):
        # # source
        # source = ""
        # for i in range(1, batch.src.size(0)):
            # sym = TEXT.vocab.itos[batch.src.data[i, 0]]
            # print("Batch.src.data ", batch.src.data, batch.src.data[i, 0])
            # if sym == "</s>": break
            # source += sym + " "
        # source += '\n'
        # if '<unk>' in source: continue

        # # target 
        # target = ""
        # for i in range(1, batch.trg.size(0)):
            # sym = TEXT.vocab.itos[batch.trg.data[i, 0]]
            # if sym == "</s>": break
            # target += sym + " "
        # target += '\n'
        # if '<unk>' in target: continue

        # # translation 
        # src_mask = (src != TEXT.vocab.stoi["<blank>"]).unsqueeze(-2)

        # if 'elmo' in EMB:
            # sentences = []
            # for i in range(len(src)):
                # sentences.append([TEXT.vocab.itos[id.item()] for id in src[i]])
            # src = batch_to_ids(sentences).to(device)
            # print(sentences)
        # print(src.shape, src_mask.shape)

        # out = greedy_decode(model, src, src_mask, 
                            # max_len=60, start_symbol=TEXT.vocab.stoi["<s>"])
        # pred = ""
        # for i in range(1, out.size(1)):
            # sym = TEXT.vocab.itos[out[0, i]]
            # if sym == "</s>": break
            # pred += sym + " "
        # pred += '\n'

        # print("Source:", source, end='')
        # print("Target:", target, end='')
        # print("Translation:", pred)
        # f_src.write(source)
        # f_trg.write(target)
        # f_pred.write(pred)

    # f_src.close()
    # f_trg.close()
    # f_pred.close()

if __name__ == "__main__":
    main()
