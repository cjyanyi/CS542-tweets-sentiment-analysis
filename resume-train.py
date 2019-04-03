# -*- coding:UTF-8 -*-
import os
import numpy as np
from params import params
from utils import load_dict,read_file
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from Attention import Attention

from utils import train_plot
from text_to_np import preprocess


BASE_DIR = ''
MODEL_DIR = os.path.join(BASE_DIR,'hierarchical_attention/')
FOLDER_DIR = os.path.join(BASE_DIR, '')
MAX_SEQUENCE_LENGTH = params['MAX_SEQUENCE_LENGTH']
MAX_NUM_WORDS = params['MAX_NUM_WORDS']
MODEL_NAME = params['MODEL_NAME']
dict_name = params['DICT_NAME']
VALIDATION_SPLIT = params['VALIDATION_SPLIT']

class Trainer:
    def __init__(self, model_name=MODEL_NAME):
        try:
            self.model = load_model(os.path.join(MODEL_DIR, model_name),custom_objects={'Attention': Attention})
            self.dict_w = load_dict(os.path.join(BASE_DIR, dict_name))
        except Exception as e:
            print(e.message)
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH


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

    def _categorical(self, labels):
        labels = to_categorical(np.asarray(labels))
        print('Shape of label tensor:', labels.shape)
        return labels

    def _split(self, data, classes):
        # split the data into a training set and a validation set
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = classes[indices]
        num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]

        return (x_train,y_train),(x_val,y_val)

    def train(self,texts,lables):
        data = self._words2seq(texts)
        classes = self._categorical(lables)
        train, val = self._split(data,classes)

        # save weights each epoch
        # filepath='weights.{epoch:02d-{val_acc:.2f}}.hdf5'
        checkpoint = ModelCheckpoint(filepath='hierarchical_attention/weights_retrain.ep{epoch:02d}-acc{val_acc:.3f}.hdf5',
                                     monitor='val_acc', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1)
        csv_logger = CSVLogger('re_training_history.log', append=False)

        history = self.model.fit(train[0], train[1],
                            batch_size=128,
                            epochs=50,
                            validation_data=(val[0], val[1]),
                            callbacks=[checkpoint, reduce_lr, early_stopping, csv_logger])

        # save the trained model
        self.model.save('hierarchical_attention/' + 'retrain-' +params['MODEL_NAME'])
        # restore the model
        # keras.models.load_model(filepath)
        train_plot(history)

if __name__ == '__main__':
    from text_to_np import read_data
    x,y = read_data()
    print(len(x))
    Trainer().train(x,y)
