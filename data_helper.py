import os
import torch
import csv
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from textblob import TextBlob
from textblob import Word
import pandas as pd
import traceback
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle


class DataHelper(object):
    def __init__(self, dataset, mode='train', vocab=None):
        allowed_data = ['imdb', 'subj', 'tweets']

        if dataset not in allowed_data:
            raise ValueError('currently allowed data: %s' % ','.join(allowed_data))
        else:
            self.dataset = dataset

        self.mode = mode

        self.base = os.path.join('data', self.dataset)


        if self.dataset == 'imdb':
            self.labels_str = ['positive','negative']
        elif self.dataset == 'subj' or self.dataset == 'tweets':
            self.labels_str = [0, 1]

        content, label, dep = self.get_content()

        self.dep = dep

        self.label = self.label_to_onehot(label)
        if vocab is None:
            self.vocab = []

            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

    def label_to_onehot(self, label_str):

        return [self.labels_str.index(l) for l in label_str]


    def get_content(self):
        if self.mode == 'test':
            data_path = os.path.join(self.base, 'test.csv')
        elif self.mode == 'train':
            data_path = os.path.join(self.base, 'train.csv')
        elif self.mode == 'dev':
            data_path = os.path.join(self.base, 'valid.csv')
        assert os.path.exists(data_path)
        data_temp = pd.read_csv(data_path, encoding='gb18030')
        data_temp.index = range(len(data_temp))
        data = data_temp[['review', 'sentiment']]

        data.index = range(len(data))
        content = list(data['review'])
        label = list(data['sentiment'])
        dep = list(data_temp['dependency'])


        return content, label, dep


    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def get_vocab(self):
        with open(os.path.join(self.base, 'vocab-5.txt')) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def build_vocab(self, content, min_count=10):
        if self.dataset != 'imdb':
            vocab = []

            for c in content:
                words = c.split(' ')
                for word in words:
                    if word not in vocab:
                        vocab.append(word)

            # 计算单词出现的频率，如果单词出现频率低于最小值，那么不将它放入单词表
            freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

            for c in content:
                words = c.split(' ')
                for word in words:
                    freq[word] += 1

            results = []
            for word in freq.keys():
                if freq[word] < min_count:
                    continue
                else:
                    results.append(word)

            results.insert(0, 'UNK')
            results.insert(0, 'PAD')
            with open(os.path.join(self.base, 'vocab-5.txt'), 'w') as f:
                f.write('\n'.join(results))

            self.vocab = results
        else:
            vocab = []

            for c in content:
                words = c.split(' ')
                for word in words:
                    if word not in vocab:
                        vocab.append(word)

            # 计算单词出现的频率，如果单词出现频率低于最小值，那么不将它放入单词表
            freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

            for c in content:
                words = c.split(' ')
                for word in words:
                    freq[word] += 1

            results = []
            for word in freq.keys():
                if freq[word] < min_count:
                    continue
                else:
                    results.append(word)

            results.insert(0, 'UNK')
            with open(os.path.join(self.base, 'vocab-5.txt'), 'w') as f:
                f.write('\n'.join(results))

            self.vocab = results


    def count_word_freq(self, content):
        freq = dict(zip(self.vocab, [0 for i in range(len(self.vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        with open(os.path.join(self.base, 'freq.csv'), 'w') as f:
            writer = csv.writer(f)
            results = list(zip(freq.keys(), freq.values()))
            writer.writerows(results)

    def get_review(self):
        return self.content, self.label, self.dep


    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]
                dep = self.dep[start:end]

                yield content, torch.tensor(label).cuda(),dep ,i


