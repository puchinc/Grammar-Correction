#!/bin/bash
#bash grammar_conda.sh --encoder models/with_error_tag.encoder \
#    --decoder models/with_error_tag.decoder \
#    --sentences CoNLL_data/train.txt --emb CoNLL_data/train_small.elmo

# for training in conda environment (i.e. lab machine)

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--encoder)
    ENCODER="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--decoder)
    DECODER="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--sentences)
    SENTENCES="$2"
    shift # past argument
    shift # past value
    ;;
    -m|--emb)
    EMB="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set - "${POSITIONAL[@]}" # restore positional parameters

# create or retrieve elmo embedding
if [[ $EMB == *"elmo"* ]]; then 
    cd ..
    source activate allennlp
    cd Grammar-Correction
    python elmo.py $SENTENCES $EMB
    conda deactivate
fi 

# train 
cd ..
source activate torch
cd Grammar-Correction
python train.py $ENCODER $DECODER $SENTENCES $EMB
conda deactivate

