#!/bin/bash
#bash grammar.sh --encoder models/with_error_tag.encoder \
#    --decoder models/with_error_tag.decoder \
#    --sentences CoNLL_data/train.txt --emb CoNLL_data/train_small.elmo

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

# train
source ../allennlp/bin/activate
python elmo.py $SENTENCES $EMB
deactivate

source ../torch/bin/activate
python train.py $ENCODER $DECODER $SENTENCES $EMB
deactivate

#echo $ENCODER
#echo $DECODER
#echo $SENTENCES
#echo $EMB

# test
