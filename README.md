# LSTM-GCN

Before you train the model, use the pre-process.py to process the data.

`python pre-process.py`

When training the model, you should pass some parameters which can be checked with the following command.

`python train.py -h`

Examples: 
`python train.py --dataset imdb --ngram 3`

`python train.py --dataset subj --ngram 3`

`python train.py --dataset tweets --ngram 4`

Note: We use the pre-trained Glove model file, `glove.6B.300d.vec.txt` which can be obtained from https://nlp.stanford.edu/projects/glove/. You need to put it in the root directory.
