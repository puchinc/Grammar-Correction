# reference
# https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb

# need to install spacy, python -m spacy download en_core_web_lg, torch, datetime
'''
Train and validate:
python seq2seq.py \
    -train_src ./data/lang8_english_src_500k.txt \
    -train_tgt ./data/lang8_english_tgt_500k.txt \
    -val_src ./data/lang8_english_src_val_100k.txt \
    -val_tgt ./data/lang8_english_src_val_100k.txt \
    -emb_type glove
'''

from Model import *
import os
import subprocess
import codecs
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import spacy
nlp = spacy.load('en_core_web_lg') # For the glove embeddings
import pickle
from allennlp.modules.elmo import Elmo, batch_to_ids
import h5py    
import codecs
from tqdm import tqdm
from collections import Counter, namedtuple
from torch.utils.data import Dataset, DataLoader

PAD = 0
BOS = 1
EOS = 2
UNK = 3

def parse_args():
    parser = argparse.ArgumentParser()        
    parser.add_argument('-train_src')                   
    parser.add_argument('-train_tgt')
    parser.add_argument('-val_src')
    parser.add_argument('-val_tgt')  
    parser.add_argument('-emb_type')               
    args = parser.parse_args()                                      
    return args

def save_model(prefix, train_dataset, encoder, decoder, opts):
    pickle.dump(train_dataset,open(prefix + 'train_dataset.pkl','wb'))
    pickle.dump(encoder,open(prefix + 'encoder.pkl','wb'))
    pickle.dump(decoder,open(prefix + 'decoder.pkl','wb'))
    pickle.dump(opts,open(prefix + 'opts.pkl','wb'))

def main():
    # parse arguments
    args = parse_args()    
    train_src = args.train_src
    train_tgt = args.train_tgt
    val_src = args.val_src
    val_tgt = args.val_tgt
    emb_type = args.emb_type

    # load data 
    train_dataset = NMTDataset(src_path=train_src,
                           tgt_path=train_tgt)
    valid_dataset = NMTDataset(src_path=val_src,
                           tgt_path=val_tgt,
                           src_vocab=train_dataset.src_vocab,
                           tgt_vocab=train_dataset.tgt_vocab)
    
    batch_size = 48
    
    train_iter = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=collate_fn)

    valid_iter = DataLoader(dataset=valid_dataset,
                        batch_size=batch_size, 
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)
    # If enabled, load checkpoint.
    LOAD_CHECKPOINT = False
    # initialize checkpoint
    init = 0
    checkpoint = {
        'opts': init,
        'global_step': init,
        'encoder_state_dict': init,
        'decoder_state_dict': init,
        'encoder_optim_state_dict': init,
        'decoder_optim_state_dict': init
    }
    if LOAD_CHECKPOINT:
        # Modify this path.
        checkpoint_path = './checkpoints/seq2seq_2019-03-01 15:19:39_acc_0.00_loss_16.69_step_10410.pt'
        checkpoint = load_checkpoint(checkpoint_path)
        opts = checkpoint['opts']    
    else:
        opts = AttrDict()

        # Configure models
        opts.word_vec_size = 300
        opts.rnn_type = 'LSTM'
        opts.hidden_size = 512
        opts.num_layers = 2
        opts.dropout = 0.3
        opts.bidirectional = True
        opts.attention = True
        opts.share_embeddings = True
        opts.pretrained_embeddings = emb_type
        opts.fixed_embeddings = True
        opts.tie_embeddings = True # Tie decoder's input and output embeddings

        # Configure optimization
        opts.max_grad_norm = 2
        opts.learning_rate = 0.001
        opts.weight_decay = 1e-5 # L2 weight regularization

        # Configure training
        opts.max_seq_len = 100 # max sequence length to prevent OOM.
        opts.num_epochs = 5
        opts.print_every_step = 20
        opts.save_every_step = 5000
	
    print('='*100)
    print('Options log:')
    print('- Load from checkpoint: {}'.format(LOAD_CHECKPOINT))
    if LOAD_CHECKPOINT: print('- Global step: {}'.format(checkpoint['global_step']))
    for k,v in opts.items(): print('- {}: {}'.format(k, v))
    print('='*100 + '\n')
        
    # Initialize vocabulary size.
    src_vocab_size = len(train_dataset.src_vocab.token2id)
    tgt_vocab_size = len(train_dataset.tgt_vocab.token2id)

    # Initialize embeddings.
    # We can actually put all modules in one module like `NMTModel`)
    # See: https://github.com/spro/practical-pytorch/issues/34
    if opts.pretrained_embeddings=='glove':
        word_vec_size = nlp.vocab.vectors_length
    elif opts.pretrained_embeddings=='elmo_input' or opts.pretrained_embeddings=='elmo_both':
        word_vec_size = 1024
    else:
        word_vec_size = opts.word_vec_size 
    
    src_embedding = nn.Embedding(src_vocab_size, word_vec_size, padding_idx=PAD)
    tgt_embedding = nn.Embedding(tgt_vocab_size, word_vec_size, padding_idx=PAD)

    if opts.share_embeddings:
        assert(src_vocab_size == tgt_vocab_size)
        tgt_embedding.weight = src_embedding.weight

    # Initialize models.
    encoder = EncoderRNN(embedding=src_embedding,
                             rnn_type=opts.rnn_type,
                             hidden_size=opts.hidden_size,
                             num_layers=opts.num_layers,
                             dropout=opts.dropout,
                             bidirectional=opts.bidirectional)

    decoder = LuongAttnDecoderRNN(encoder, embedding=tgt_embedding,
                                      attention=opts.attention,
                                      tie_embeddings=opts.tie_embeddings,
                                      dropout=opts.dropout)
    if opts.pretrained_embeddings=='glove':
        glove_embeddings = load_spacy_glove_embedding(nlp, train_dataset.src_vocab)
        encoder.embedding.weight.data.copy_(glove_embeddings)
        decoder.embedding.weight.data.copy_(glove_embeddings)
        if opts.fixed_embeddings:
            encoder.embedding.weight.requires_grad = False
            decoder.embedding.weight.requires_grad = False
    elmo = None
    if opts.pretrained_embeddings == 'elmo_input' or opts.pretrained_embeddings == 'elmo_both':
        options_file = '../data/embs/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        weight_file = '../data/embs/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        elmo = Elmo(options_file, weight_file, 1, dropout=0)
    if LOAD_CHECKPOINT:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Move models to GPU (need time for initial run)
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
            
    FINE_TUNE = True
    if FINE_TUNE:
        encoder.embedding.weight.requires_grad = True  
            
    print('='*100)
    print('Model log:\n')
    print(encoder)
    print(decoder)
    print('- Encoder input embedding requires_grad={}'.format(encoder.embedding.weight.requires_grad))
    print('- Decoder input embedding requires_grad={}'.format(decoder.embedding.weight.requires_grad))
    print('- Decoder output embedding requires_grad={}'.format(decoder.W_s.weight.requires_grad))
    print('='*100 + '\n')
        
    # Initialize optimizers (we can experiment different learning rates)
    encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay)
    decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay)
    

    # 1) training
    training(encoder, decoder, encoder_optim, decoder_optim, train_iter, valid_iter, opts, load_checkpoint, checkpoint, elmo)
     
    print('Done training. Start Validation.')
    save_model('', train_dataset, encoder, decoder, opts) 
    # 2) validation 
    total_loss = 0
    total_corrects = 0
    total_words = 0

    for batch_id, batch_data in tqdm(enumerate(valid_iter)):
        src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data
        max_seq_len = max(src_lens + tgt_lens)
        if max_seq_len > opts.max_seq_len:
            print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, opts.max_seq_len))
            continue
        loss, pred_seqs, attention_weights, num_corrects, num_words \
                = evaluate(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, decoder, opts, elmo)

        total_loss += loss
        total_corrects += num_corrects
        total_words += num_words
        total_accuracy = 100 * (total_corrects / total_words)

    print('='*100)
    print('Validation log:')
    print('- Total loss: {}'.format(total_loss))
    print('- Total corrects: {}'.format(total_corrects))
    print('- Total words: {}'.format(total_words))
    print('- Total accuracy: {}'.format(total_accuracy))
    print('='*100 + '\n')
    
    # save model
    save_model('',train_dataset, encoder, decoder, opts)        

if __name__ == '__main__':
    main()
