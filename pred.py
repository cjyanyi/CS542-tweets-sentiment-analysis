# -*- coding:UTF-8 -*-

__author__ = 'jy.cai'

from keras.models import load_model
import os
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

from params import params
from utils import load_dict,read_file
from text_to_np import preprocess


BASE_DIR = ''
MODEL_DIR = os.path.join(BASE_DIR,'hierarchical_attention/')
FOLDER_DIR = os.path.join(BASE_DIR, '')
MAX_SEQUENCE_LENGTH = params['MAX_SEQUENCE_LENGTH']
MAX_NUM_WORDS = params['MAX_NUM_WORDS']
MODEL_NAME = params['MODEL_NAME']
dict_name = params['DICT_NAME']

class TweetsAnalysis(object):

    def __init__(self, model_name=MODEL_NAME, test_dir='test_tweets'):
        try:
            self.model = load_model(os.path.join(MODEL_DIR, model_name))
            self.dict_w = load_dict(os.path.join(BASE_DIR, dict_name))
        except Exception as e:
            print(e.message)
        self.test_dir = os.path.join(BASE_DIR, test_dir)
        self.labels_index = params['LABEL_INDEX']
        self.labels = ('negative','positive','neutral')
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    def read_texts(self,name):
        path = self.test_dir
        fpath = os.path.join(path, name)
        texts = []
        texts.append(read_file(fpath))
        print('Found 1 Raw text.')
        return texts


    def load_test(self,path=None):
        #path = os.path.join(BASE_DIR, 'fake')
        if path == None:
            path = self.test_dir

        #read all files in path
        texts = []
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(read_file(fpath))
        print('Found %s texts.' % len(texts))
        # print texts
        for str in texts:
            print([str[:80]])
        # covert texts to tensors
        return texts

    def texts_to_labels(self,texts, verb=False):
        data = self._words2seq(texts)
        predictions = self.predict(data)
        return (self.decode_predictions(predictions,verbose=verb))

    def _words2seq(self,texts):
        '''
        convert texts to input vectors of model
        '''
        # finally, vectorize the text samples into a 2D integer tensor
        dict = self.dict_w
        data = []
        for text in texts:
            #word split
            content = text_to_word_sequence(preprocess(text))
            one_data = []
            #word to token
            for word in content:
                if word in dict.keys() and dict[word] < MAX_NUM_WORDS:
                    one_data.append(dict[word])
                else:
                    one_data.append(0)
            data.append(one_data)

        data = np.array(data)
        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', data.shape)
        return data

    def predict(self,texts:list):
        # inference, from input vectors to probability
        predictions = self.model.predict(self._words2seq(texts))
        print('Model Predict: ', predictions)
        return predictions

    def decode_predictions(self,predictions, verbose =True):
        '''to get labels of predictions'''
        labels = []
        for pre in predictions:
            indice = 0
            if pre[1]>0.6:
                indice = 1
            elif pre[0]>0.6:
                indice = 0
            else:
                indice = 2
            labels.append(indice)
            if verbose == True:
                print('This is a {} ! ** neg: {:-4}%  and pos: {:-4}%'.format(self.labels[indice], pre[0]*100,pre[1]*100))
        return labels

    def text_preprocess(self,str):
        dict = self.dict_w
        #content = text_to_word_sequence(''.join(texts))
        content = text_to_word_sequence(preprocess(str))
        data = []
        for word in content:
            if word in dict.keys() and dict[word] < MAX_NUM_WORDS:
                data.append(dict[word])
            else:
                data.append(0)
        # length of a text,  word_seq, data input of model
        return min(len(data),self.MAX_SEQUENCE_LENGTH),content,data


if __name__ == "__main__":
        ta = TweetsAnalysis()
        #texts = ta.load_test()
        tests = ['I hate the rain']
        predictions = ta.predict(tests)
        ta.decode_predictions(predictions)