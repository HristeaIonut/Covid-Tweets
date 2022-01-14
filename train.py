import pickle as pkl
import numpy as np
from keras.models import load_model
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

data = pkl.load(open("datas/data_with_retweet.pkl", "rb"))
X, y = zip(*data)
max_length = 104

X = np.asarray(X)
y = np.asarray(y, dtype=np.float32)
old_range = 1 - (-1)
new_range = 1 - 0
y = (y - (-1)) * new_range / old_range
y = [0 if i < 0.5 else 1 for i in y]
y = np.asarray(y, dtype=np.int32)

max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
tweets = pad_sequences(sequences, maxlen=max_length)
vocab_size = len(tokenizer.word_index)+1

model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
model.fit(tweets, y, batch_size=32, epochs=10, validation_split=0.3)
model.save("model.h5")
