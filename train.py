from __future__ import unicode_literals, print_function, division
from io import open
import sys
import os
import unicodedata
import string
import re
import random
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from AttnDecoderRNN import *
from EncoderRNN import *
from Lang import *
from helper import *
from config import *

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

def tensorsFromPair(pair, if_elmo=True):
    if if_elmo:
        return pair
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def tensorsFromElmoText(pair, output_lang):
    input_tensor = pair[0].to(device)
    target_tensor = tensorFromSentence(output_lang, ' '.join(pair[1]))
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
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

def trainIters(training_pairs, encoder, decoder, n_iters, encoder_path, decoder_path, print_every=1000, plot_every=100, learning_rate=0.01):
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
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
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

def load_elmo_pairs(path):
    with open(path, 'rb') as elmo:
        return pickle.load(elmo)

if __name__ == '__main__':
    # python train.py encoder.model decoder.model train.txt train.elmo 
    small = False
    if len(sys.argv) == 5:
        encoder_path = sys.argv[1]
        decoder_path = sys.argv[2]
        sentence_path = sys.argv[3]
        emb_path = sys.argv[4]
    # Small dataset
    else:
        encoder_path = 'models/with_error_tag.encoder'
        decoder_path = 'models/with_error_tag.decoder'
        sentence_path = 'CoNLL_data/train.txt'
        emb_path = 'CoNLL_data/train_small.elmo'
    if emb_path == 'CoNLL_data/train_small.elmo':
        small = True
    # Absolute path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    encoder_path = os.path.join(dir_path, encoder_path)
    decoder_path = os.path.join(dir_path, decoder_path)
    sentence_path = os.path.join(dir_path, sentence_path)
    emb_path = os.path.join(dir_path, emb_path)

    input_lang, output_lang, indices, pairs = prepareData(sentence_path, 'wrong', 'correct', small=small)
    print(random.choice(pairs))

    elmo_pairs = load_elmo_pairs(emb_path)
    pairs = [elmo_pairs[i] for i in indices]
    training_pairs = [tensorsFromElmoText(random.choice(pairs), output_lang) for i in range(n_iters)]
    # training_pairs = [random.choice(pairs) for i in range(n_iters)]
    
    encoder = EncoderRNN(elmo_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

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
    
    trainIters(training_pairs, encoder, decoder, n_iters, encoder_path, 
            decoder_path, print_every=print_every)
