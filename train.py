import pickle

from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, TextVectorization
from models.covid_tweets import CovidTweets
import tensorflow as tf
import pickle as pkl
import numpy as np


if __name__ == '__main__':
    data = pkl.load(open("datas/data_with_retweet.pkl", "rb"))
    X, y = zip(*data)
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float32)
    old_range = 1 - (-1)
    new_range = 1 - 0
    y = (y - (-1)) * new_range / old_range
    y = [0 if i < 0.5 else 1 for i in y]
    y = np.asarray(y, dtype=np.int32)

    embedding = pickle.load(open('models/embedding.layer', 'rb'))

    model = tf.keras.Sequential([
        TextVectorization(
            vocabulary=pickle.load(open('datas/vocabulary.pkl', 'rb')),
            output_mode='int',
            output_sequence_length=60
        ),
        embedding,
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # model = CovidTweets('datas/metadata.tsv', 'datas/vectors.tsv')
    model.compile(loss='mse',
             optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, batch_size=128, epochs=20, validation_split=0.2)
    model.save("model.h5")
