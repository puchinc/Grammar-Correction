from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

from seq2seq.AttnDecoderRNN import *
from seq2seq.EncoderRNN import *
from seq2seq.config import *
import pickle

# CONSTANT
SOS_token = config["SOS_token"]
EOS_token = config["EOS_token"]
MAX_LENGTH = config["MAX_LENGTH"]

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, sentence, input_lang, output_lang,  max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, sentence_pairs, elmo_pairs, input_lang, output_lang, n=10):
    for i in range(n):
        sentence_pair = random.choice(sentence_pairs)
        print('>', sentence_pair[0])
        print('=', sentence_pair[1])
        output_words, attentions = evaluate(encoder, decoder, sentence_pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
        # calculate bleu score
        tokens_reference = word_tokenize(sentence_pair[1])
        tokens_target = word_tokenize(output_sentence)
        print(tokens_reference)
        print(tokens_target)
        score = sentence_bleu([tokens_reference], tokens_target)
        print(score)

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(encoder, decoder, sentence, input_lang, output_lang,  max_length=MAX_LENGTH):
    output_words, attentions = evaluate(encoder, decoder, sentence, input_lang, output_lang,  max_length=MAX_LENGTH)
    print('input =', sentence)
    print('output =', ' '.join(output_words))
    showAttention(sentence, output_words, attentions)

def load_variables():
    encoder = pickle.load(open('encoder.pkl','rb'))
    decoder = pickle.load(open('decoder.pkl','rb'))
    sentence_pairs = pickle.load(open('sentence_pairs.pkl', 'rb'))
    pairs = pickle.load(open('pairs.pkl', 'rb'))
    input_lang = pickle.load(open('input_lang.pkl', 'rb'))
    output_lang = pickle.load(open('output_lang.pkl', 'rb'))
    return encoder, decoder, sentence_pairs, pairs, input_lang, output_lang

# TODO
def main():
    encoder, decoder, sentence_pairs, pairs, input_lang, output_lang = load_variables()

    # translation/evaluate
    # works for nn.embedding. TODO for elmo, bert
    evaluateRandomly(encoder, decoder, sentence_pairs, pairs, input_lang, output_lang)
    # evaluateAndShowAttention(encoder, decoder, 'here i want to share forest view on this issue .', input_lang, output_lang, max_length=MAX_LENGTH)
     
if __name__ == '__main__':
    main()
