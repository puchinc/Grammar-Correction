# Seq2seq Neural Grammar Correction

The goal of this project is to experiment with elmo embedding and to see if there's an improvement when we change the embedding to elmo for sequence-to-sequence translation for grammar correction. 

## Requirements
All dependencies can be installed via:
```
pip install -r requirements.txt
```

### Virtualenv

You can also use virtualenv, which needs two virtualenvs named allennlp and torch. allennlp is for elmo embedding, and torch is for machine translation. GPU is required for generating new elmo embeddings and Python3 is used.

[elmo] https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md 
* allennlp

        pip install allennlp

[NMT] https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* torch

        pip install matplotlib
        pip install torch==0.4.0 
        pip install torchvision


### Dataset 

The dataset are from CoNLL-2013 and CoNLL-2014 Shared Task for grammar correction. They have original sentence and corrected sentence with position of error in the sentence and error type. CoNLL-2013 has 5 types of errors while CoNLL-2014 has 28 types of errors. 


## Quickstart

### Step 1: Preprocess the data
```
python lang8_parser.py -i data/src/lang-8-20111007-L1-v2.dat -o data/src -l2 English

```

### Step 2: Train the model
```
python train.py \
    -i data/test/conll.txt \
    -e data/test/conll.elmo \
    -enc data/test/with_error_tag.encoder \
    -dec data/test/with_error_tag.decoder
```

### Step 3: Correct the grammar
TODO

