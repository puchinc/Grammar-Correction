#!/usr/bin/python

# Example Usage:
#   python train.py \
#       -i data/test/conll.txt \
#       -e data/test/conll.elmo \
#       -enc data/test/with_error_tag.encoder \
#       -dec data/test/with_error_tag.decoder

from __future__ import unicode_literals, print_function, division
from io import open
import sys
import os
import unicodedata
import string
import re
import random
import pickle
import time
import math
import argparse
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Lang import Lang

from seq2seq.AttnDecoderRNN import AttnDecoderRNN
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.config import config

# CONSTANT
SOS_token = config["SOS_token"]
EOS_token = config["EOS_token"]
MAX_LENGTH = config["MAX_LENGTH"]

def parse_args():
    parser = argparse.ArgumentParser()        
    parser.add_argument('-i', '--input_file')                   
    parser.add_argument('-e', '--embedding')
    parser.add_argument('-enc', '--encoder_path')
    parser.add_argument('-dec', '--decoder_path')
    parser.add_argument('-o', '--output_dir')                   
    args = parser.parse_args()                                      

    return args

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

def readLangs(path, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path, encoding='utf-8').read().strip().split('\n')
   
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    idx, pair = zip(*[(i, pair) for i, pair in enumerate(pairs) if filterPair(pair)]) 
    return list(idx), list(pair)

def prepareData(path, lang1, lang2, reverse=False, small=False):
    input_lang, output_lang, pairs = readLangs(path, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if small:
        indices = [i for i in range(100)]
    else:
        indices, pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, indices, pairs

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def loadConll(path='CoNLL_data/train.txt'):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[[e for e in normalizeString(s).split(' ')] for s in l.split('\t')] for l in lines]
    return pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang, if_elmo=True):
    #if if_elmo:
    #    return pair
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def tensorsFromElmoText(pair, output_lang):
    input_tensor = pair[0].to(device)
    target_tensor = tensorFromSentence(output_lang, ' '.join(pair[1]))
    return (input_tensor, target_tensor)

def tensorsFromBertText(pair, output_lang):
    input_tensor = pair[0].to(device)
    target_tensor = tensorFromSentence(output_lang, ' '.join(pair[1]))
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device) 
    loss = 0 
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device) 
    # decoder_input = torch.zeros([1, 1024], device=device)
    decoder_hidden = encoder_hidden 
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(int(target_length/2)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(training_pairs, encoder, decoder, n_iters, encoder_path, decoder_path, teacher_forcing_ratio, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)

            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def tensorsToDevice(pair):
    input_tensor = pair[0].to(device)
    target_tensor = pair[1].to(device)
    return (input_tensor, target_tensor)

def load_elmo_pairs(path):
    with open(path, 'rb') as elmo:
        return pickle.load(elmo)

def load_bert_pairs(path):
    with open(path, 'rb') as bert:
        return pickle.load(bert)

def save_variables(encoder, decoder, sentence_pairs, pairs, input_lang, output_lang, nn_embedding, emb_type):
    pickle.dump(encoder,open(emb_type+'encoder.pkl','wb'))
    pickle.dump(decoder,open(emb_type+'decoder.pkl','wb'))
    pickle.dump(sentence_pairs,open(emb_type+'sentence_pairs.pkl','wb'))
    pickle.dump(pairs,open(emb_type+'pairs.pkl','wb'))
    pickle.dump(input_lang,open(emb_type+'input_lang.pkl','wb'))
    pickle.dump(output_lang,open(emb_type+'output_lang.pkl','wb'))
    pickle.dump(nn_embedding, open(emb_type+'nn_embedding.pkl','wb'))

def main():
    args = parse_args()
    encoder_path = args.encoder_path
    decoder_path = args.decoder_path
    sentence_path = args.input_file
    emb_path = args.embedding
    emb_type = ""

    hidden_size = config["hidden_size"]
    elmo_size = config["elmo_size"]
    bert_size = config["bert_size"]

    # TRAIN
    teacher_forcing_ratio = config["teacher_forcing_ratio"]
    print_every = config["print_every"]
    n_iters = config["n_iters"]

    small = False
    nn_embedding = False

    if 'elmo' in emb_path:
        small = True
        emb_type = "elmo_"
    elif 'bert' in emb_path:
        big = True
        emb_type = "bert_"
    elif emb_path == 'nn.embedding':
        nn_embedding = True 

    # Absolute path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    encoder_path = os.path.join(dir_path, encoder_path)
    decoder_path = os.path.join(dir_path, decoder_path)
    sentence_path = os.path.join(dir_path, sentence_path)
    emb_path = os.path.join(dir_path, emb_path)

    input_lang, output_lang, indices, pairs = prepareData(sentence_path, 'wrong', 'correct', small=small)
    print(random.choice(pairs))

    sentence_pairs = pairs    
    if 'elmo' in emb_path:
        elmo_pairs = load_elmo_pairs(emb_path)
        pairs = [elmo_pairs[i] for i in indices]
    if 'bert' in emb_path:
        bert_pairs = load_bert_pairs(emb_path)
        pairs = [bert_pairs[i] for i in indices]

    if nn_embedding:
        training_pairs = [tensorsFromPair(random.choice(sentence_pairs), input_lang, output_lang) for i in range(n_iters)]
    elif small == True:
        training_pairs = [tensorsFromElmoText(random.choice(pairs), output_lang) for i in range(n_iters)]
    elif big == True:
        training_pairs = [tensorsFromBertText(random.choice(pairs), output_lang) for i in range(n_iters)]
    else:
    	training_pairs = [tensorsToDevice(random.choice(pairs)) for i in range(n_iters)]

    if nn_embedding:
        # use nn.embedding
        encoder = EncoderRNN(input_lang.n_words, hidden_size, 'nn.embedding').to(device)
    elif small == True:
        # use elmo embedding
        encoder = EncoderRNN(elmo_size, hidden_size).to(device)
    elif big == True:
        # use bert embedding
        encoder = EncoderRNN(bert_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    ''' 
    if os.path.isfile(encoder_path) and os.path.isfile(decoder_path):
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
        # use cpu to load gpu trained models
        else:
            encoder.load_state_dict(torch.load(encoder_path,
                    map_location=lambda storage, loc: storage))
            decoder.load_state_dict(torch.load(decoder_path, 
                    map_location=lambda storage, loc: storage))
    '''
    trainIters(training_pairs, encoder, decoder, n_iters, encoder_path, 
            decoder_path, teacher_forcing_ratio, print_every=print_every)
    
    save_variables(encoder, decoder, sentence_pairs, pairs, input_lang, output_lang, nn_embedding, emb_type) 

if __name__ == '__main__':
    main()
