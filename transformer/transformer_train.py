# Reference: 
# codebase: http://nlp.seas.harvard.edu/2018/04/03/attention.html
# torchtext load pretrained embeddings: http://anie.me/On-Torchtext/

# Prelims:
# pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
# python -m spacy download en 

# Train:
# python annotated_transformer.py

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

from Model import MyIterator, LabelSmoothing, NoamOpt, MultiGPULossCompute, SimpleLossCompute, Batch, Elmo_Batch
from Model import build_model, make_model, run_epoch, rebatch, batch_size_fn, elmo_rebatch

def main():
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    # EMB_DIM should be multiple of 8, look at MultiHeadedAttention
    # EMB = 'bow'
    EMB = 'elmo'
    # EMB = 'glove.6B.200d'
    EMB_DIM = 512
    BATCH_SIZE = 1000
    EPOCHES = 3

    # GPU to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = ("cpu")
    devices = [0, 1, 2, 3]

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    src_dir = os.path.join(root_dir, 'data/src')
    test_dir = os.path.join(root_dir, 'data/test')
    eval_dir = os.path.join(root_dir, 'data/eval')
    model_file = os.path.join(root_dir, 'data/models', EMB + '.transformer.pt')
    elmo_options_file = os.path.join(root_dir, 'data/embs/elmo.json')
    elmo_weights_file = os.path.join(root_dir, 'data/embs/elmo.hdf5')
    vocab_file = os.path.join(root_dir, 'data/models', 'english.vocab')

    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
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

    train = datasets.TranslationDataset(path=os.path.join(src_dir, 'lang8.train'),
            exts=('.src', '.trg'), fields=(TEXT, TEXT))
    val = datasets.TranslationDataset(path=os.path.join(src_dir, 'lang8.val'), 
            exts=('.src', '.trg'), fields=(TEXT, TEXT))

    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    random_idx = random.randint(0, len(train) - 1)
    print(train[random_idx].src)
    print(train[random_idx].trg)

    #####################
    #   Word Embedding  #
    #####################

    # weights = None
    emb = None
    data_generator = lambda data: data

    # glove embedding
    if 'glove' in EMB:
        TEXT.build_vocab(train.src, vectors=EMB)
        pad_idx = TEXT.vocab.stoi["<blank>"]

        EMB_DIM = 200
        data_generator = lambda data: (rebatch(pad_idx, b) for b in data)
        emb = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

    # TODO elmo embedding
    elif 'elmo' in EMB: 
        from allennlp.modules.elmo import Elmo
        elmo = Elmo(elmo_options_file, elmo_weights_file, 1, dropout=0).to(device)

        MIN_FREQ = 2
        TEXT.build_vocab(train.src, min_freq=MIN_FREQ)
        pad_idx = TEXT.vocab.stoi["<blank>"]

        EMB_DIM = 1024
        data_generator = lambda data: (elmo_rebatch(pad_idx, b, TEXT.vocab, device) for b in data)
        emb = lambda ids: elmo(ids)['elmo_representations'][0]

    # TODO bert embedding
    elif 'bert' in EMB: pass
    else:
        MIN_FREQ = 2
        TEXT.build_vocab(train.src, min_freq=MIN_FREQ)
        pad_idx = TEXT.vocab.stoi["<blank>"]

        data_generator = lambda data: (rebatch(pad_idx, b) for b in data)
        emb = nn.Embedding(len(TEXT.vocab), EMB_DIM)

    # train_iter = batch * sen * word
    # for b in train:
        # print(b)
        # sys.exit()

    # for b in train_iter:
        # for sen in b.src:
            # for word_id in sen:
                # print(TEXT.vocab.itos[word_id.item()], end=' ')
            # print()
        # sys.exit()

    # for b in data.Iterator(train, batch_size=2, device=device):
        # for sen in b.src:
            # for word_id in sen:
                # print(TEXT.vocab.itos[word_id.item()], end=' ')
            # print()
        # sys.exit()

    # print("Save Vocabuary...")
    # torch.save(TEXT.vocab, vocab_file)


    # TODO torchtext load customized pretrained weights
    ##########################
    #   Training the System  #
    ##########################
    # model = make_model(len(TEXT.vocab), d_model=EMB_DIM, emb_weight=weights, N=6)
    model = build_model(len(TEXT.vocab), emb, d_model=EMB_DIM).to(device)
    criterion = LabelSmoothing(size=len(TEXT.vocab), padding_idx=pad_idx, smoothing=0.1).to(device)

    model_opt = NoamOpt(EMB_DIM, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, 
                        betas=(0.9, 0.98), eps=1e-9))

    ### MULTIPLE GPU
    # model_par = nn.DataParallel(model, device_ids=devices)
    # for epoch in range(EPOCHES):
        # model_par.train()
        
        # run_epoch(data_generator(train_iter), model_par, 
                  # MultiGPULossCompute(model.generator, criterion, devices, opt=model_opt))
        # print("Model saved...")
        # torch.save(model.state_dict(), model_file)

        # model_par.eval()
        # loss = run_epoch(data_generator(valid_iter), model_par, 
                         # MultiGPULossCompute(model.generator, criterion, devices, opt=None))
        # print("Epoch %d/%d - Loss: %f" % (epoch + 1, EPOCHES, loss))

    ### SINGLE GPU
    for epoch in range(EPOCHES):
        model.train()
        run_epoch(data_generator(train_iter), 
                  model, 
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
        print("Model saved...")
        torch.save([model.state_dict(), TEXT.vocab], model_file)

        model.eval()
        loss = run_epoch(data_generator(valid_iter), 
                          model, 
                          SimpleLossCompute(model.generator, criterion, opt=None))
        print("Epoch %d/%d - Loss: %f" % (epoch + 1, EPOCHES, loss))


if __name__ == "__main__":
    main()
