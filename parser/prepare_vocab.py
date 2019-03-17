import torch
from torchtext import data, datasets
import spacy
import sys
import os

def main():
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    DATA = 'conll'

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    src_dir = os.path.join(root_dir, 'data/src')
    vocab_file = os.path.join(root_dir, 'data/models', '%s.vocab' % (DATA))
    vocab_freq_file = os.path.join(root_dir, 'data/src', '%s.vocab.txt' % (DATA))

    spacy_en = spacy.load('en')
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)


    ###############
    #  Vocabuary  #
    ###############
    if os.path.exists(vocab_file):
        print("Building vocabuary...")
        TEXT.vocab = torch.load(vocab_file)
    else:
        print("Loading data...")
        train = datasets.TranslationDataset(path=os.path.join(src_dir, 
            DATA), exts=('.train.src', '.train.trg'), fields=(TEXT, TEXT))
        MIN_FREQ = 2
        TEXT.build_vocab(train.src, min_freq=MIN_FREQ)

    #########################
    #  Save in count order  #
    #########################

    ordered_words = [word for word, _ in TEXT.vocab.freqs.most_common()]
    with open(vocab_freq_file, 'w') as f:
        print('Writing...')
        f.write('<S>\n</S>\n<UNK>\n')

        for word in ordered_words:
            f.write(word + '\n')
        
if __name__ == "__main__":
    main()
