import pickle as pkl
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import tensorflow.ragged as ragged
from keras.layers import Input, Embedding, LSTM, Dense, Activation
from tensorflow.python.ops.ragged.ragged_tensor import Ragged

import tensorflow as tf

data = pkl.load(open("data_with_retweet.pkl", "rb"))
X, y = zip(*data)
test = X[0:3]
X = np.asarray(X)
y = np.asarray(y)
max_features = max(len(x) for x in X)
r_X = ragged.constant(X)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
    tf.keras.layers.Embedding(max_features,128),
    tf.keras.layers.LSTM(32, use_bias=False),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation(tf.nn.relu),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(r_X, y, batch_size=32, epochs=10, validation_split=0.1)
