# -*- coding:UTF-8 -*-
import os

params = {
    'MAX_SEQUENCE_LENGTH': 150,
    'MAX_NUM_WORDS':40000,
    'EMBEDDING_DIM': 200,
    'VALIDATION_SPLIT':0.2,
    'LABEL_INDEX': {'negative':0, 'positive':1},
    'BASE_DIR':'',
    'MODEL_DIR':os.path.join('','hierarchical_attention/'),
    'MODEL_NAME':'lstm-attention-model.h5',
    'DICT_NAME':'dict_tweets.json'
}