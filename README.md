# Seq2seq Neural Grammar Correction

The goal of this project is to experiment with elmo and bert embedding along with transformer framework and to see if there's an improvement for grammar correction. 

## Requirements

### Dataset 

Two datasets
1. CoNLL-2013 and CoNLL-2014 Shared Task for grammar correction. They have original sentence and corrected sentence with position of error in the sentence and error type. CoNLL-2013 has 5 types of errors while CoNLL-2014 has 28 types of errors. 
2. Lang8

### Virtualenv

You need three virtualenvs named allennlp, torch, and transformer\_env. allennlp is for the ELMo embedding, torch is for machine translation, and transformer\_env is for the Transformer model. GPU is required for generating new elmo embeddings and Python3 is used.

[ELMo] https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md 
* allennlp

        pip install allennlp

[NMT] https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* torch

        pip install torch
        pip install torchvision
        pip install matplotlib

[Transformer] http://nlp.seas.harvard.edu/2018/04/03/attention.html 
* transformer\_env

        pip install -r transformer/transformer_requirements.txt

## Quickstart

### Step 1: Preprocess the data
```
python lang8_parser.py \
    -i lang-8-20111007-L1-v2.dat \
    -o data/src \
    -l2 English
```
### Step 2: Pretrained word embeddings
```
(allennlp_venv)
python emb/elmo.py data/test/conll.txt data/embeddings/conll.elmo 
```

### Step 3: Train the model
```
(torch_venv)
python train.py \
    -i data/test/conll.txt \
    -e data/test/conll.elmo \
    -enc data/test/with_error_tag.encoder \
    -dec data/test/with_error_tag.decoder
```

### Step 4: Correct the grammar
TODO

