## CS249 Final Project

### Goal

The goal of this project is to experiment with elmo embedding and to see if there's an improvement when we change the embedding to elmo for sequence-to-sequence translation for grammar correction. 

### Dataset 

The dataset are from CoNLL-2013 and CoNLL-2014 Shared Task for grammar correction. They have original sentence and corrected sentence with position of error in the sentence and error type. CoNLL-2013 has 5 types of errors while CoNLL-2014 has 28 types of errors. 

### Virtualenv

You need two virtualenvs, one for elmo embedding, and the other for machine translation.

[elmo] https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md 
* allennlp

        pip install allennlp
        pip install 

[NMT] https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* torch

        pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl 
        pip3 install torchvision
