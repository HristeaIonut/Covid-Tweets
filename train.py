import pickle as pkl
from keras.backend import backend
from keras.optimizer_experimental import sgd
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
import tensorflow as tf
from keras import optimizers


data = pkl.load(open("data_with_retweet.pkl", "rb"))
X, y = zip(*data)
max_length = 104

X = np.asarray(X)
y = np.asarray(y, dtype=np.float32)
old_range = 1 - (-1)
new_range = 1 - 0
y = (y - (-1)) * new_range / old_range
y = [0 if i < 0.5 else 1 for i in y]
y = np.asarray(y, dtype=np.int32)
# print(y[0:20])

max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
tweets = pad_sequences(sequences, maxlen=max_length)
vocab_size = len(tokenizer.word_index)+1


# model = Sequential()
# model.add(Embedding(20000, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
# model.fit(tweets, y, batch_size=32, epochs=10)
# model.save("model.h5")
model = load_model("model.h5")
res = model.predict(tweets)
for i in range(20):
    print(np.round(res[i]), y[i])