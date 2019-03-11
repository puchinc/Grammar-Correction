# Seq2seq Neural Grammar Correction

The goal of this project is to experiment with elmo and bert embedding along with transformer framework and to see if there's an improvement for grammar correction. 

## Requirements

Three datasets
1. CoNLL-2013 and CoNLL-2014 Shared Task for grammar correction. They have original sentence and corrected sentence with position of error in the sentence and error type. CoNLL-2013 has 5 types of errors while CoNLL-2014 has 28 types of errors. 
2. Lang8
3. AESW Dataset 


### Step 1.1: Preprocess conll
remove words between <del></del>, then remove all tags, trim leading and unnecessary spaces 

```
awk -F $'\t' '{print $1}' data/src/conll.txt | perl -i -pe 's|<del>.*?</del>||g' | perl -i -pe 's|<.*?>||g' | sed -e 's/^[ \t]*//' | tr -s ' ' > data/src/conll.src
awk -F $'\t' '{print $2}' data/src/conll.txt | perl -i -pe 's|<del>.*?</del>||g' | perl -i -pe 's|<.*?>||g' | sed -e 's/^[ \t]*//' | tr -s ' ' > data/src/conll.trg
```

### Step 1.2: Preprocess conll2014
remove empty lines, lines contain http, tag, strange long character words, and very short sentences; trim leading and unnecessary spaces 

```
grep -vwE "(http|<.*>|^[[:space:]]*$|\w{20,}|^.{0,50}$)" data/conll/conll2014_allerrors.txt > data/src/conll2014.txt

awk -F $'\t' '{print $1}' data/src/conll2014.txt > data/src/conll2014.src  
awk -F $'\t' '{print $2}' data/src/conll2014.txt > data/src/conll2014.trg  
```

### Step 1.3: Preprocess lang8
```
python parser/lang8_parser.py \
       -i lang-8-20111007-L1-v2.dat \
       -o data/src \
       -l2 English
awk -F $'\t' '{print $1}' data/src/lang8.txt > data/src/lang8.src 
awk -F $'\t' '{print $2}' data/src/lang8.txt > data/src/lang8.trg
```

### Step 2: Split datasets
```
cd data/src
python ../../parser/prepare_csv.py \
    -i conll2014.src \
    -train conll2014.train.src \
    -train_r 0.6 \
    -test conll2014.test.src \
    -test_r 0.2 \
    -val conll2014.val.src \
    -val_r 0.2
python ../../parser/prepare_csv.py \
    -i conll2014.trg \
    -train conll2014.train.trg \
    -train_r 0.6 \
    -test conll2014.test.trg \
    -test_r 0.2 \
    -val conll2014.val.trg \
    -val_r 0.2
cd -
```

### Step 3: Download pretrained word embeddings
```
wget -P data/embs/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget -P data/embs/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
```

### Step 4: Virtualenv

[Transformer] http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html
[ELMo] https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md 
* transformer\_env

        pip install allennlp
        pip install torch numpy matplotlib spacy torchtext seaborn 
        python -m spacy download en 

[Batched seq2seq] https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
* batched\_seq2seq\_env

        pip install -r batched_seq2seq/requirements.txt
        python -m spacy download en_core_web_lg
    
[BERT] https://github.com/huggingface/pytorch-pretrained-BERT
* bert
        
        pip install pytorch-pretrained-bert

## Transformer Quickstart

### Step 1: Train the model
```
(transformer_env)
CUDA_VISIBLE_DEVICES=0,1,2,3 python transformer/trainsformer_train.py
```

### Step 2: Evaluate the model
```
(transformer_env)
python evaluation/gleu.py \
    -s data/eval/conll.glove.glove.eval.src \
    -r data/eval/conll.glove.glove.eval.trg \
    --hyp data/eval/conll.glove.glove.eval.pred
``` 

## Batched Seq2seq Quickstart

### Step 1: Train and validate the model

You can replace the arguments with source and target of your choice. `emb_type` can be: `glove` for GloVe embedding, `none` for no pretrained embedding, `elmo_input` for ELMo embedding for input, and `elmo_both` for ELMo embedding for both input and output. 

```
cd batched_seq2seq
(batched_seq2seq_env)
python seq2seq_train.py \
       -train_src ./data/lang8_english_src_10k.txt \
       -train_tgt ./data/lang8_english_tgt_10k.txt \
       -val_src ./data/source.txt \
       -val_tgt ./data/target_valid.txt \
       -emb_type glove
```

### Step 2: Test the model
```
(batched_seq2seq_env)
python seq2seq_pred.py \
       -test_src ./data/source_test.txt
```

### Step 3: Evaluate the model
```
(batched_seq2seq_env)
python ../evaluation/gleu.py \
       -s ./data/source_test.txt \
       -r ./data/target_test1.txt \
       --hyp ./data/pred.txt

```
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

## Fine tuning ELMo Model on new data
[Tensorflow]
* tf

        pip install tensorflow tensorflow-gpu==1.2 h5py
        python setup.py install
        python -m unittest discover tests/

[Elmo-Tutorial] https://github.com/PrashantRanjan09/Elmo-Tutorial
[BiLM-TF] https://github.com/allenai/bilm-tf
* tf
    
        wget -P data/embs/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.data-00000-of-00001
        wget -P data/embs/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.index
        wget -P data/embs/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.meta
        wget -P data/embs/ https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/options.json

        python parser/prepare_vocab.py

        python bilm-tf/bin/dump_weights.py \
            --save_dir data/embs \
            --outfile data/embs/weights.hdf5

        export CUDA_VISIBLE_DEVICES=2,3,4
        python bilm-tf/bin/restart.py \
                --train_prefix='data/src/lang8.train.*' \
                --vocab_file=data/src/vocab-2016-09-10.txt \
                --save_dir=data/embs \
                --n_gpus=3 \
                --batch_size=3000 \
                --n_epochs=10

        python bilm-tf/bin/train_elmo.py \
                --train_prefix='data/src/lang8.train.*' \
                --vocab_file data/src/vocab-2016-09-10.txt \
                --save_dir data/embs/

        export CUDA_VISIBLE_DEVICES=0,1,2
        python bilm-tf/bin/train_elmo_updated.py \
                --train_prefix='data/src/lang8.train.*' \
                --vocab_file=data/src/lang8.vocab.txt \
                --save_dir=data/embs \
                --restart_ckpt_file=data/embs/checkpoint

        
