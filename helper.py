from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Lang import *
from config import *

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
        indices = [i for i in range(10)]
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


import time
import math


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


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


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
