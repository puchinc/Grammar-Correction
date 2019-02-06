#!/bin/bash

# TODO
# use python train.py rather than bash

# bash grammar.sh \
#    --emb data/CoNLL_data/train_small.elmo \
#    --encoder data/models/with_error_tag.encoder \
#    --decoder data/models/with_error_tag.decoder \
#    --sentences data/CoNLL_data/train.txt          

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

# check null
if [ -z $ENCODER ]; then 
    echo "Missing Arguments"
    exit 1
fi
if [ -z $DECODER ]; then 
    echo "Missing Arguments"
    exit 1
fi
if [ -z $SENTENCES ]; then 
    echo "Missing Arguments"
    exit 1
fi
if [ -z $EMB ]; then 
    echo "Missing Arguments"
    exit 1
fi

# training in conda environment if conda exists (i.e. lab machine)
if [ -x "$(command -v conda)" ]; then 
    # create or retrieve elmo embedding
    if [[ $EMB == *"elmo"* ]]; then 
        cd ..
        source activate allennlp
        cd Grammar-Correction
        python elmo.py $SENTENCES $EMB
        conda deactivate
    fi 

    cd ..
    source activate torch
    cd Grammar-Correction
    python train.py $ENCODER $DECODER $SENTENCES $EMB
    conda deactivate
# for training in local environment 
else
    # train
    if [[ $EMB == *"elmo"* ]]; then
        #source ../allennlp/bin/activate
        python emb/elmo.py $SENTENCES $EMB
        #deactivate
    fi

    #source ../torch/bin/activate
    python train.py $ENCODER $DECODER $SENTENCES $EMB
    #deactivate
fi


