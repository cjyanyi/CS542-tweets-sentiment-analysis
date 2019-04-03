# -*- coding:UTF-8 -*-
'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Dropout, GRU, BatchNormalization,Flatten,Activation
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.callbacks import ModelCheckpoint
from utils import create_path


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'enron')
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

nb_filters = 128
hiden_lstm_layer = 160

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {'ham':0, 'spam':1}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        #label_id = len(labels_index)
        #labels_index[name] = label_id
        for folder in sorted(os.listdir(path)):
            sndpath = os.path.join(path, folder)
            if os.path.isdir(sndpath):
                for fname in sorted(os.listdir(sndpath)):
                    if fname is not None:
                        fpath = os.path.join(sndpath, fname)
                        args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                        with open(fpath, **args) as f:
                            t = f.read()
                            i = t.find('Subject:')  # skip header
                            if 0 < i:
                                t = t[i:]
                            texts.append(t)
                        labels.append(0) if folder =='ham' else labels.append(1)

#print (labels[:5000])

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')
create_path('lstm_kmax_softmax')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# train a 1D convnet with global maxpooling
x = Dropout(0.2)(embedded_sequences)
x = Conv1D(nb_filters, 3, padding='valid')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(5,padding='same')(x)
x = Conv1D(nb_filters, 3, activation="relu", padding='valid')(x)
x = MaxPooling1D(5,padding='same')(x)
x = GRU(hiden_lstm_layer, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
x = MaxPooling1D(10,padding='same')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
preds = Dense(len(labels_index), activation='softmax')(x)
#rmsprop = RMSprop(lr=0.001)

model = Model(sequence_input, preds)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
               )
# save weights each epoch
#filepath='weights.{epoch:02d-{val_acc:.2f}}.hdf5'
checkpoint = ModelCheckpoint(filepath='lstm_kmax_softmax/weights.ep{epoch:02d}-acc{val_acc:.3f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val),
          callbacks = [checkpoint])

#save the trained model
model.save('lstm_kmax_softmax/lstm-kmax-model.h5')
#restore the model
#keras.models.load_model(filepath)