# transformer using allennlp
# reference: https://github.com/mhagiwara/realworldnlp/blob/master/examples/mt/mt.py
# http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html
# inputs: lang8_small_train.csv and lang8_small_val.csv with source\ttarget 
# requirements: pip install allennlp 
# to get train and val files:
# python prepare_csv.py \
#       -i ../data/test/lang8_small.txt \
#       -train ../data/test/lang8_small_train.csv \
#       -train_r 0.6 \
#       -test ../data/test/lang8_small_test.csv \
#       -test_r 0.2 \
#       -val ../data/test/lang8_small_val.csv \
#       -val_r 0.2

import itertools
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer

EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0

def main():
    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('../data/test/lang8_small_train.csv')
    validation_dataset = reader.read('../data/test/lang8_small_val.csv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    
    encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})
   
    attention = DotProductAttention()

    max_decoding_steps = 20
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True)
    model.to(torch.device('cuda'), torch.float)
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=1,
                      cuda_device=CUDA_DEVICE)

    for i in range(50):
        print('Epoch: {}'.format(i))
        trainer.train()

        predictor = SimpleSeq2SeqPredictor(model, reader)

        for instance in itertools.islice(validation_dataset, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])


if __name__ == '__main__':
    main()
