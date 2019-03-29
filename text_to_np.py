# -*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
import os

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
# decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#stop_words = stopwords.words("english")
stop_words = set(['a','as','the','t','s','at','just','such','an','so'])
stemmer = SnowballStemmer("english")

import re

def read_data():
    if not os.path.exists('./processed/texts.csv'):
        return read_csv()
    text_df = pd.read_csv('./processed/texts.csv',header=None)
    label_df = pd.read_csv('./processed/labels.csv',header=None)
    print ('first 20 labels: ',label_df[0].to_list()[:20])
    return text_df[0].astype('str').to_list(),label_df[0].to_list()

def read_csv():
    dataset_filename = os.listdir("../dataset")[0]
    dataset_path = os.path.join("..", "dataset", dataset_filename)
    print("Open file:", dataset_path)
    df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    print("Dataset size:", len(df))
    print(df.head(10))
    df.text = df.text.apply(lambda x: preprocess(x))
    df.target = df.target.apply(lambda x:x//4)
    df.text.to_csv('./processed/texts.csv',header=None, index=None)
    df.target.to_csv('./processed/labels.csv',header=None, index=None)
    return df.text.tolist(), df.target.tolist()



def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens) if len(tokens)!=0 else 'null'

#read_data()

