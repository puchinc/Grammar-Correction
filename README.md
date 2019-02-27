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
        pip install scipy

[Transformer] http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html
* transformer\_env

        pip install allennlp

[Batched seq2seq] https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
* batched\_seq2seq\_env

        pip install -r batched_seq2seq/requirements.txt
        python -m spacy download en_core_web_lg
    
[BERT] https://github.com/huggingface/pytorch-pretrained-BERT
* bert
        
        pip install pytorch-pretrained-bert
        
## BERT Embedding

### Train word embeddings
```
(bert_venv)
python emb/bert.py --input_file data/test/lang8_small.txt \
        --output_file data/embeddings/lang8_small.bert \
        --bert_mode bert-base-uncased \
        --do_lower_case \
        --batch_size 16
```
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

### Step 4: Evaluate the model
```
(torch_venv)
python translate.py
python evaluation/gleu.py  \
       -s ./source.txt \
       -r ./target.txt \
       --hyp ./pred.txt
```

## Transformer Quickstart

### Annotated Transformer

### Step 1: Preprocess the data
```
awk -F $'\t' '{print $1}' data/src/lang8.txt > data/src/lang8.src 
awk -F $'\t' '{print $2}' data/src/lang8.txt > data/src/lang8.trg
cd data/src
python ../../transformer/prepare_csv.py \
       -i lang8.src \
       -train lang8.train.src \
       -train_r 0.6 \
       -test lang8.test.src \
       -test_r 0.2 \
       -val lang8.val.src \
       -val_r 0.2
python ../../transformer/prepare_csv.py \
       -i lang8.trg \
       -train lang8.train.trg \
       -train_r 0.6 \
       -test lang8.test.trg \
       -test_r 0.2 \
       -val lang8.val.trg \
       -val_r 0.2
cd -
```

### Step 2: Train the model
```
(transformer_env)
python transformer/transformer_allennlp.py
```

### Step 3: Evaluate the model
```
(transformer_env)
python evaluation/gleu.py \
       -s data/src/lang8.val.src
       -r data/src/lang8.val.trg \
       --hyp data/src/translation.txt
``` 

---

### Allennlp Transformer

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

### Step 2: Train the model
```
(transformer_env)
python transformer_allennlp.py
```

### Step 3: Evaluate the model
```
(transformer_env)
python ../evaluation/gleu.py \
       -s ./source.txt 
       -r ./target.txt \
       --hyp ./pred.txt
``` 

## Batched Seq2seq Quickstart

### Step 1: Train the model
```
cd batched_seq2seq
(batched_seq2seq_env)
python seq2seq.py
```

### Step 2: Evaluate the model
```
(batched_seq2seq_env)
python ../evaluation/gleu.py \
       -s ./data/source_test.txt 
       -r ./data/target_test0.txt \
          ./data/target_test1.txt \
          ./data/target_test2.txt \
          ./data/target_test3.txt \
       --hyp ./data/pred.txt
```

