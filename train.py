import pickle as pkl
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

import tensorflow as tf


def letters_to_numbers(list_of_tweets):
    new_list = list()
    for i in list_of_tweets:
        tweet = list()
        for letter in i:
            tweet.append(ord(letter))
        new_list.append(tweet)
    return new_list


data = pkl.load(open("data_with_retweet.pkl", "rb"))
X, y = zip(*data)
# X = letters_to_numbers(X)
max_length = max([len(i) for i in X])
# X = [i + [0] * (max_length - len(i)) for i in X]
X = np.asarray(X)
y = np.asarray(y, dtype=np.float32)

max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
tweets = pad_sequences(sequences, maxlen=max_length)

model = Sequential()
model.add(Embedding(max_words, 64))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(tweets, y, batch_size=32, epochs=10, validation_split=0.1)
model.save("model.h5")