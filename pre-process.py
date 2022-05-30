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


tokenizer = ToktokTokenizer()

#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

m_list = ['test', 'valid', 'train']

base = '/home/xqliu/baselines/transform/tweets'

for mode in m_list:

    if mode == 'train':
        data_path = os.path.join(base, 'raw_train.csv')
    elif mode == 'test':
        data_path = os.path.join(base, 'raw_test.csv')
    elif mode == 'valid':
        data_path = os.path.join(base, 'raw_valid.csv')
    assert os.path.exists(data_path)
    data_temp = pd.read_csv(data_path, encoding='utf-8')
    data_temp.index = range(len(data_temp))
    reviews = data_temp['review']
    data = data_temp
    # Apply function on review column
    data['review'] = data['review'].apply(denoise_text)
    # Apply function on review column
    data['review'] = data['review'].apply(remove_special_characters)
    # Apply function on review column
    data['review'] = data['review'].apply(simple_stemmer)
    # Apply function on review column
    data['review'] = data['review'].apply(remove_stopwords)

    data.index = range(len(data))

    data[['review', 'sentiment']].to_csv('p'+mode+'.csv', index=False)

