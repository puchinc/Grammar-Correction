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
        pip install nltk

[Transformer] http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html
* transformer\_env

        pip install allennlp

[Batched seq2seq] https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
* batched\_seq2seq\_env

        pip install -r batched_seq2seq/requirements.txt
        python -m spacy download en_core_web_lg

## ELMo Quickstart

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
python emb/elmo.py data/test/lang8_small.txt data/embeddings/lang8_small.elmo 
```

### Step 3: Train the model
```
(torch_venv)
python train.py \
    -i data/test/lang8_small.txt \
    -e data/embeddings/lang8_small.elmo \
    -enc data/test/with_error_tag.encoder \
    -dec data/test/with_error_tag.decoder
```

### Step 4: Correct the grammar
```
(torch_venv)
python translate.py
```
## Transformer Quickstart

### Step 1: Preprocess the data
```
cd transformer
python prepare_csv.py \
       -i ../data/test/lang8_small.txt \
       -train ../data/test/lang8_small_train.csv \
       -train_r 0.6 \
       -test ../data/test/lang8_small_test.csv \
       -test_r 0.2 \
       -val ../data/test/lang8_small_val.csv \
       -val_r 0.2
```

### Step 2: Train and evaluate the model
```
(transformer_env)
python transformer_allennlp.py
```
### Batched Seq2seq Quickstart

### Step 1: Train the model
```
cd batched_seq2seq
(batched_seq2seq_env)
python seq2seq.py
```

### Step 2: Evaluate the model
```
(batched_seq2seq_env)
python ./data/gleu.py \
       -s ./data/source_test.txt 
       -r ./data/target_test0.txt \
          ./data/target_test1.txt \
          ./data/target_test2.txt \
          ./data/target_test3.txt \
       --hyp ./data/pred.txt
```

