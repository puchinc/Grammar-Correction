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
import sys
from allennlp.modules.elmo import Elmo, batch_to_ids

nlp = spacy.load('en_core_web_lg') # For the glove embeddings

""" Enable GPU training """
USE_CUDA = torch.cuda.is_available()
print('Use_CUDA={}'.format(USE_CUDA))
if USE_CUDA:
    # You can change device by `torch.cuda.set_device(device_id)`
    print('current_device={}'.format(torch.cuda.current_device()))
    
import codecs
from tqdm import tqdm
from collections import Counter, namedtuple
from torch.utils.data import Dataset, DataLoader

PAD = 0
BOS = 1
EOS = 2
UNK = 3

class AttrDict(dict):
    """ Access dictionary keys like attribute 
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class NMTDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab=None, tgt_vocab=None, max_vocab_size=50000, share_vocab=True):
        """ Note: If src_vocab, tgt_vocab is not given, it will build both vocabs.
            Args: 
            - src_path, tgt_path: text file with tokenized sentences.
            - src_vocab, tgt_vocab: data structure is same as self.build_vocab().
        """
        print('='*100)
        print('Dataset preprocessing log:')
        
        print('- Loading and tokenizing source sentences...')
        self.src_sents = self.load_sents(src_path)
        print('- Loading and tokenizing target sentences...')
        self.tgt_sents = self.load_sents(tgt_path)
        
        if src_vocab is None or tgt_vocab is None:
            print('- Building source counter...')
            self.src_counter = self.build_counter(self.src_sents)
            print('- Building target counter...')
            self.tgt_counter = self.build_counter(self.tgt_sents)

            if share_vocab:
                print('- Building source vocabulary...')
                self.src_vocab = self.build_vocab(self.src_counter + self.tgt_counter, max_vocab_size)
                print('- Building target vocabulary...')
                self.tgt_vocab = self.src_vocab
            else:
                print('- Building source vocabulary...')
                self.src_vocab = self.build_vocab(self.src_counter, max_vocab_size)
                print('- Building target vocabulary...')
                self.tgt_vocab = self.build_vocab(self.tgt_counter, max_vocab_size)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            share_vocab = src_vocab == tgt_vocab
                        
        print('='*100)
        print('Dataset Info:')
        print('- Number of source sentences: {}'.format(len(self.src_sents)))
        print('- Number of target sentences: {}'.format(len(self.tgt_sents)))
        print('- Source vocabulary size: {}'.format(len(self.src_vocab.token2id)))
        print('- Target vocabulary size: {}'.format(len(self.tgt_vocab.token2id)))
        print('- Shared vocabulary: {}'.format(share_vocab))
        print('='*100 + '\n')
    
    def __len__(self):
        return len(self.src_sents)
    
    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token2id, append_BOS=False, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.token2id, append_BOS=False, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq
    
    def load_sents(self, file_path):
        sents = []
        with codecs.open(file_path) as file:
            for sent in tqdm(file.readlines()):
                tokens = [token for token in sent.split()]
                sents.append(tokens)
        return sents
    
    def build_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter.update(sent)
        return counter
    
    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
        vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v:k for k,v in tqdm(vocab.token2id.items())}    
        return vocab
    
    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq
    
def collate_fn(data):
    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    
    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)
       
    """
    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs = zip(*data)
    
    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)
    
    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0,1)
    tgt_seqs = tgt_seqs.transpose(0,1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens
  
class EncoderRNN(nn.Module):
    def __init__(self, embedding=None, rnn_type='LSTM', hidden_size=128, num_layers=1, dropout=0.3, bidirectional=True):
        super(EncoderRNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions
        
        self.embedding = embedding
        self.word_vec_size = self.embedding.embedding_dim
        
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
                           input_size=self.word_vec_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=self.dropout, 
                           bidirectional=self.bidirectional)
        
    def forward(self, src_seqs, src_lens, elmo_emb=None, hidden=None):
        """
        Args:
            - src_seqs: (max_src_len, batch_size)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        
        # (max_src_len, batch_size) => (max_src_len, batch_size, word_vec_size)
        if elmo_emb is None:
            emb = self.embedding(src_seqs)
        else:
            emb = elmo_emb

        # packed_emb:
        # - data: (sum(batch_sizes), word_vec_size)
        # - batch_sizes: list of batch sizes
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, src_lens)

        # rnn(gru) returns:
        # - packed_outputs: shape same as packed_emb
        # - hidden: (num_layers * num_directions, batch_size, hidden_size) 
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        # output_lens == src_lensˇ
        outputs, output_lens =  nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size) 
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)
        
        return outputs, hidden
    
    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)
            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)
            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)
            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)
            
        return hidden
      
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, encoder, embedding=None, attention=True, bias=True, tie_embeddings=False, dropout=0.3):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025
            
            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462
        """
        super(LuongAttnDecoderRNN, self).__init__()
        
        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        self.embedding = embedding
        self.attention = attention
        self.tie_embeddings = tie_embeddings
        
        self.vocab_size = self.embedding.num_embeddings
        self.word_vec_size = self.embedding.embedding_dim
        
        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
                            input_size=self.word_vec_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout)
        
        if self.attention:
            self.W_a = nn.Linear(encoder.hidden_size * encoder.num_directions,
                                 self.hidden_size, bias=bias)
            self.W_c = nn.Linear(encoder.hidden_size * encoder.num_directions + self.hidden_size, 
                                 self.hidden_size, bias=bias)
        
        if self.tie_embeddings:
            self.W_proj = nn.Linear(self.hidden_size, self.word_vec_size, bias=bias)
            self.W_s = nn.Linear(self.word_vec_size, self.vocab_size, bias=bias)
            self.W_s.weight = self.embedding.weight
        else:
            self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)
        
    def forward(self, input_seq, decoder_hidden, encoder_outputs, src_lens):
        """ Args:
            - input_seq      : (batch_size)
            - decoder_hidden : (t=0) last encoder hidden state (num_layers * num_directions, batch_size, hidden_size) 
                               (t>0) previous decoder hidden state (num_layers, batch_size, hidden_size)
            - encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        
            Returns:
            - output           : (batch_size, vocab_size)
            - decoder_hidden   : (num_layers, batch_size, hidden_size)
            - attention_weights: (batch_size, max_src_len)
        """        
        # (batch_size) => (seq_len=1, batch_size)
        input_seq = input_seq.unsqueeze(0)
        
        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size) 
        emb = self.embedding(input_seq)
        
        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)
        decoder_output, decoder_hidden = self.rnn(emb, decoder_hidden)

        # (seq_len=1, batch_size, hidden_size) => (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0,1)
        
        """ 
        ------------------------------------------------------------------------------------------
        Notes of computing attention scores
        ------------------------------------------------------------------------------------------
        # For-loop version:
        max_src_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attention_scores = Variable(torch.zeros(batch_size, max_src_len))
        # For every batch, every time step of encoder's hidden state, calculate attention score.
        for b in range(batch_size):
            for t in range(max_src_len):
                # Loung. eq(8) -- general form content-based attention:
                attention_scores[b,t] = decoder_output[b].dot(attention.W_a(encoder_outputs[t,b]))
        ------------------------------------------------------------------------------------------
        # Vectorized version:
        1. decoder_output: (batch_size, seq_len=1, hidden_size)
        2. encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        3. W_a(encoder_outputs): (max_src_len, batch_size, hidden_size)
                        .transpose(0,1)  : (batch_size, max_src_len, hidden_size) 
                        .transpose(1,2)  : (batch_size, hidden_size, max_src_len)
        4. attention_scores: 
                        (batch_size, seq_len=1, hidden_size) * (batch_size, hidden_size, max_src_len) 
                        => (batch_size, seq_len=1, max_src_len)
        """
        
        if self.attention:
            # attention_scores: (batch_size, seq_len=1, max_src_len)
            attention_scores = torch.bmm(decoder_output, self.W_a(encoder_outputs).transpose(0,1).transpose(1,2))

            # attention_mask: (batch_size, seq_len=1, max_src_len)
            attention_mask = sequence_mask(src_lens).unsqueeze(1)

            # Fills elements of tensor with `-float('inf')` where `mask` is 1.
            attention_scores.data.masked_fill_(1 - attention_mask.data, -float('inf'))

            # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax` 
            # => (batch_size, seq_len=1, max_src_len)
            try: # torch 0.3.x
                attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)
            except:
                attention_weights = F.softmax(attention_scores.squeeze(1)).unsqueeze(1)

            # context_vector:
            # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
            # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
            context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0,1))

            # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
            concat_input = torch.cat([context_vector, decoder_output], -1)

            # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)
            concat_output = F.tanh(self.W_c(concat_input))
            
            # Prepare returns:
            # (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len)
            attention_weights = attention_weights.squeeze(1)
        else:
            attention_weights = None
            concat_output = decoder_output
        
        # If input and output embeddings are tied,
        # project `decoder_hidden_size` to `word_vec_size`.
        if self.tie_embeddings:
            output = self.W_s(self.W_proj(concat_output))
        else:
            # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)
            output = self.W_s(concat_output)    
        
        # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)
        
        del src_lens
        
        return output, decoder_hidden, attention_weights

# glove embedding
def load_spacy_glove_embedding(spacy_nlp, vocab):
    
    vocab_size = len(vocab.token2id)
    word_vec_size = spacy_nlp.vocab.vectors_length
    embedding = np.zeros((vocab_size, word_vec_size))
    unk_count = 0
    
    print('='*100)
    print('Loading spacy glove embedding:')
    print('- Vocabulary size: {}'.format(vocab_size))
    print('- Word vector size: {}'.format(word_vec_size))
    
    for token, index in tqdm(vocab.token2id.items()):
        if token == vocab.id2token[PAD]: 
            continue
        elif token in [vocab.id2token[BOS], vocab.id2token[EOS], vocab.id2token[UNK]]: 
            vector = np.random.rand(word_vec_size,)
        elif spacy_nlp.vocab[token].has_vector: 
            vector = spacy_nlp.vocab[token].vector
        else:
            vector = embedding[UNK] 
            unk_count += 1
            
        embedding[index] = vector
        
    print('- Unknown word count: {}'.format(unk_count))
    print('='*100 + '\n')
        
    return torch.from_numpy(embedding).float()

# elmo embedding 
def load_elmo_embeddings(sentences, max_seq):
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    character_ids = batch_to_ids(sentences)
    elmo_embeddings = elmo(character_ids)['elmo_representations'][0]
    
    embeddings = elmo_embeddings.transpose(0,1)
    # dimension: (max_src_len, batch_size, word_vec_size)
    emb_zeros = torch.zeros(max_seq-embeddings.size()[0], embeddings.size()[1], embeddings.size()[2])
    embeddings = torch.cat((embeddings, emb_zeros), 0)
    return embeddings

def sequence_mask(sequence_length, max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand
    
    return mask

def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
        
    The code is same as:
    
    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    
    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0,1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)
    
    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words

def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

def save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    total_accuracy, total_loss, global_step):
    checkpoint = {
        'opts': opts,
        'global_step': global_step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }
    
    checkpoint_path = 'checkpoints/%s_acc_%.2f_loss_%.2f_step_%d.pt' % (experiment_name, total_accuracy, total_loss, global_step)
    
    directory, filename = os.path.split(os.path.abspath(checkpoint_path))

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path

def variable2numpy(var):
    """ For tensorboard visualization """
    return var.data.cpu().numpy()

def write_to_tensorboard(writer, global_step, total_loss, total_corrects, total_words, total_accuracy,
                         encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm,
                         encoder, decoder, gpu_memory_usage=None):
    # scalars
    if gpu_memory_usage is not None:
        writer.add_scalar('curr_gpu_memory_usage', gpu_memory_usage['curr'], global_step)
        writer.add_scalar('diff_gpu_memory_usage', gpu_memory_usage['diff'], global_step)
        
    writer.add_scalar('total_loss', total_loss, global_step)
    writer.add_scalar('total_accuracy', total_accuracy, global_step)
    writer.add_scalar('total_corrects', total_corrects, global_step)
    writer.add_scalar('total_words', total_words, global_step)
    writer.add_scalar('encoder_grad_norm', encoder_grad_norm, global_step)
    writer.add_scalar('decoder_grad_norm', decoder_grad_norm, global_step)
    writer.add_scalar('clipped_encoder_grad_norm', clipped_encoder_grad_norm, global_step)
    writer.add_scalar('clipped_decoder_grad_norm', clipped_decoder_grad_norm, global_step)
    
    # histogram
    for name, param in encoder.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram('encoder/{}'.format(name), variable2numpy(param), global_step, bins='doane')
        if param.grad is not None:
            writer.add_histogram('encoder/{}/grad'.format(name), variable2numpy(param.grad), global_step, bins='doane')

    for name, param in decoder.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram('decoder/{}'.format(name), variable2numpy(param), global_step, bins='doane')
        if param.grad is not None:
            writer.add_histogram('decoder/{}/grad'.format(name), variable2numpy(param.grad), global_step, bins='doane')
            
def detach_hidden(hidden):
    """ Wraps hidden states in new Variables, to detach them from their history. Prevent OOM.
        After detach, the hidden's requires_grad=Fasle and grad_fn=None.
    Issues:
    - Memory leak problem in LSTM and RNN: https://github.com/pytorch/pytorch/issues/2198
    - https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    - https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
    - https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
    - 
    """
    if type(hidden) == Variable:
        hidden.detach_() # same as creating a new variable.
    else:
        for h in hidden: h.detach_()

def get_gpu_memory_usage(device_id):
    """Get the current gpu usage. """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map[device_id]
def compute_grad_norm(parameters, norm_type=2):
    """ Ref: http://pytorch.org/docs/0.3.0/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens,
          encoder, decoder, encoder_optim, decoder_optim, opts):    
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert(batch_size == tgt_seqs.size(1))
    
    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs)
    tgt_seqs = Variable(tgt_seqs)
    src_lens = Variable(torch.LongTensor(src_lens))
    tgt_lens = Variable(torch.LongTensor(tgt_lens))

    # Decoder's input
    input_seq = Variable(torch.LongTensor([BOS] * batch_size))
    
    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()
    
    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = Variable(torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size))

    # pretrained embedding
    if opts.pretrained_embeddings == 'elmo':
        elmo_emb = load_elmo_embeddings(src_sents, src_seqs.size()[0])
    else:
        elmo_emb = None

    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        tgt_seqs = tgt_seqs.cuda()
        src_lens = src_lens.cuda()
        tgt_lens = tgt_lens.cuda()
        input_seq = input_seq.cuda()
        decoder_outputs = decoder_outputs.cuda()
        if elmo_emb is not None:
            elmo_emb = elmo_emb.cuda()
        
    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    encoder.train()
    decoder.train()
    
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
        
    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist(), elmo_emb)

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden
    
    # Run through decoder one time step at a time.
    for t in range(max_tgt_len):
        
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output
        
        # Next input is current target
        input_seq = tgt_seqs[t]
        
        # Detach hidden state:
        detach_hidden(decoder_hidden)
        
    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss, pred_seqs, num_corrects, num_words = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0,1).contiguous(), 
        tgt_seqs.transpose(0,1).contiguous(),
        tgt_lens
    )
    
    pred_seqs = pred_seqs[:max_tgt_len]
    
    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()
    
    # Clip gradients
    encoder_grad_norm = nn.utils.clip_grad_norm(encoder.parameters(), opts.max_grad_norm)
    decoder_grad_norm = nn.utils.clip_grad_norm(decoder.parameters(), opts.max_grad_norm)
    clipped_encoder_grad_norm = compute_grad_norm(encoder.parameters())
    clipped_decoder_grad_norm = compute_grad_norm(decoder.parameters())
    
    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()
        
    return loss.item(), pred_seqs, attention_weights, num_corrects, num_words,\
           encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm

def training(encoder, decoder, encoder_optim, decoder_optim, train_iter, valid_iter, opts, load_checkpoint, checkpoint):
    """ Open port 6006 and see tensorboard.
    Ref:  https://medium.com/@dexterhuang/%E7%B5%A6-pytorch-%E7%94%A8%E7%9A%84-tensorboard-bb341ce3f837
    """
    from datetime import datetime
    # from tensorboardX import SummaryWriter
    # --------------------------
    # Configure tensorboard
    # --------------------------
    model_name = 'seq2seq'
    datetime = ('%s' % datetime.now()).split('.')[0]
    experiment_name = '{}_{}'.format(model_name, datetime)
    #tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
    #writer = SummaryWriter(tensorboard_log_dir)

    # --------------------------
    # Configure training
    # --------------------------
    num_epochs = opts.num_epochs
    print_every_step = opts.print_every_step
    save_every_step = opts.save_every_step
    # For saving checkpoint and tensorboard
    global_step = 0 if not load_checkpoint else checkpoint['global_step']

    # --------------------------
    # Start training
    # --------------------------
    total_loss = 0
    total_corrects = 0
    total_words = 0
    prev_gpu_memory_usage = 0
     
    for epoch in range(num_epochs):
        for batch_id, batch_data in tqdm(enumerate(train_iter)):

            # Unpack batch data
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data
            # Ignore batch if there is a long sequence.
            max_seq_len = max(src_lens + tgt_lens)
            if max_seq_len > opts.max_seq_len:
                print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, opts.max_seq_len))
                continue

            # Train.
            loss, pred_seqs, attention_weights, num_corrects, num_words, \
            encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm \
            = train(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, decoder, encoder_optim, decoder_optim, opts)

            # Statistics.
            global_step += 1
            total_loss += loss
            total_corrects += num_corrects
            total_words += num_words
            total_accuracy = 100 * (total_corrects / total_words)

            # Save checkpoint.
            if global_step % save_every_step == 0:

                checkpoint_path = save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim, 
                                                  total_accuracy, total_loss, global_step)

                print('='*100)
                print('Save checkpoint to "{}".'.format(checkpoint_path))
                print('='*100 + '\n')

            # Print statistics and write to Tensorboard.
            if global_step % print_every_step == 0:

                curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
                diff_gpu_memory_usage = curr_gpu_memory_usage - prev_gpu_memory_usage
                prev_gpu_memory_usage = curr_gpu_memory_usage

                print('='*100)
                print('Training log:')
                print('- Epoch: {}/{}'.format(epoch, num_epochs))
                print('- Global step: {}'.format(global_step))
                print('- Total loss: {}'.format(total_loss))
                print('- Total corrects: {}'.format(total_corrects))
                print('- Total words: {}'.format(total_words))
                print('- Total accuracy: {}'.format(total_accuracy))
                print('- Current GPU memory usage: {}'.format(curr_gpu_memory_usage))
                print('- Diff GPU memory usage: {}'.format(diff_gpu_memory_usage))
                print('='*100 + '\n')

                # write_to_tensorboard(writer, global_step, total_loss, total_corrects, total_words, total_accuracy,
                #                     encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm,
                #                     encoder, decoder,
                #                     gpu_memory_usage={
                #                         'curr': curr_gpu_memory_usage,
                #                         'diff': diff_gpu_memory_usage
                #                     })

                total_loss = 0
                total_corrects = 0
                total_words = 0

            # Free memory
            del src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, \
                loss, pred_seqs, attention_weights, num_corrects, num_words, \
                encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm
    checkpoint_path = save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim, 
                                              total_accuracy, total_loss, global_step)
            
    print('='*100)
    print('Save checkpoint to "{}".'.format(checkpoint_path))
    print('='*100 + '\n')

def evaluate(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, decoder, opts):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert(batch_size == tgt_seqs.size(1))
    
    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs, volatile=True)
    tgt_seqs = Variable(tgt_seqs, volatile=True)
    src_lens = Variable(torch.LongTensor(src_lens), volatile=True)
    tgt_lens = Variable(torch.LongTensor(tgt_lens), volatile=True)

    # Decoder's input
    input_seq = Variable(torch.LongTensor([BOS] * batch_size), volatile=True)
    
    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()
    
    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = Variable(torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size), volatile=True)

    if opts.pretrained_embeddings == 'elmo':
        elmo_emb = load_elmo_embeddings(src_sents, src_seqs.size()[0])
    else:
        elmo_emb = None
    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        tgt_seqs = tgt_seqs.cuda()
        src_lens = src_lens.cuda()
        tgt_lens = tgt_lens.cuda()
        input_seq = input_seq.cuda()
        decoder_outputs = decoder_outputs.cuda()
        if elmo_emb is not None:
            elmo_emb = elmo_emb.cuda()
    # -------------------------------------
    # Evaluation mode (disable dropout)
    # -------------------------------------
    encoder.eval()
    decoder.eval()
            
    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist(), elmo_emb)
    
    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden
    #if max_tgt_len > opts.max_seq_len:
    #    max_tgt_len = opts.max_seq_len 
    # Run through decoder one time step at a time.
    for t in range(max_tgt_len):
        
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output
        
        # Next input is current target
        input_seq = tgt_seqs[t]
        
        # Detach hidden state (may not need this, since no BPTT)
        detach_hidden(decoder_hidden)
        
    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss, pred_seqs, num_corrects, num_words = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0,1).contiguous(), 
        tgt_seqs.transpose(0,1).contiguous(),
        tgt_lens
    )
    
    pred_seqs = pred_seqs[:max_tgt_len]
    
    return loss.item(), pred_seqs, attention_weights, num_corrects, num_words

def translate(src_text, train_dataset, encoder, decoder, opts, max_seq_len, replace_unk=True):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Like dataset's `__getitem__()` and dataloader's `collate_fn()`.
    src_sent = src_text.split()
    src_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=src_text.split(),
                                                          token2id=train_dataset.src_vocab.token2id,
                                                          append_BOS=False, append_EOS=True)]).transpose(0,1)
    src_lens = [len(src_seqs)]
    
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    
    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs, volatile=True)
    src_lens = Variable(torch.LongTensor(src_lens), volatile=True)

    # Decoder's input
    input_seq = Variable(torch.LongTensor([BOS] * batch_size), volatile=True)
    # Store output words and attention states
    out_sent = []
    all_attention_weights = torch.zeros(max_seq_len, len(src_seqs))
   
    # pretrained embedding
    if opts.pretrained_embeddings == 'elmo':
        elmo_emb = load_elmo_embeddings([src_sent], src_seqs.size()[0])
    else:
        elmo_emb = None 
    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        src_lens = src_lens.cuda()
        input_seq = input_seq.cuda()
        if elmo_emb is not None:
            elmo_emb = elmo_emb.cuda()
        
    # -------------------------------------
    # Evaluation mode (disable dropout)
    # -------------------------------------
    encoder.eval()
    decoder.eval()
        
    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist(), elmo_emb)

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden
     
    # Run through decoder one time step at a time.
    for t in range(max_seq_len):
        
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store attention weights.
        # .squeeze(0): remove `batch_size` dimension since batch_size=1
        all_attention_weights[t] = attention_weights.squeeze(0).cpu().data 
        
        # Choose top word from decoder's output
        prob, token_id = decoder_output.data.topk(1)
        token_id = token_id[0][0] # get value
        if token_id == EOS:
            break
        else:
            if token_id == UNK and replace_unk:
                # Replace unk by selecting the source token with the highest attention score.
                score, idx = all_attention_weights[t].max(0)
                if idx.item() > len(src_sent):
                    token = src_sent[idx.item()]
                else:
                    break
            else:
                # <UNK>
                token = train_dataset.tgt_vocab.id2token[token_id.item()]
            
            out_sent.append(token)
        
        # Next input is chosen word
        input_seq = Variable(torch.LongTensor([token_id]), volatile=True)
        if USE_CUDA: input_seq = input_seq.cuda()
            
        # Repackage hidden state (may not need this, since no BPTT)
        detach_hidden(decoder_hidden)
    
    src_text = ' '.join([train_dataset.src_vocab.id2token[token_id] for token_id in src_seqs.data.squeeze(1).tolist()])
    out_text = ' '.join(out_sent)
        
    # all_attention_weights: (out_len, src_len)
    return src_text, out_text, all_attention_weights[:len(out_sent)]
