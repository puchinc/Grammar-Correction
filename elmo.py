# Need allennlp with torch==0.3.1 && GPU

from allennlp.modules.elmo import Elmo, batch_to_ids
from helper import *

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

def elmoFromPair(pair):
    # input_character_ids = batch_to_ids(pair[0])
    # input_tensor = elmo(input_character_ids)
    input_character_ids = batch_to_ids(pair[1])
    input_tensor = elmo(input_character_ids)
    output_character_ids = batch_to_ids(pair[1])
    output_tensor = elmo(output_character_ids)
    return (input_tensor, output_tensor)


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(elmoFromPair(pairs[0])) 
