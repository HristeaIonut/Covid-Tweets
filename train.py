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
    y = np.rint(y + 1)
    y = np.asarray([[1 if i == z else 0 for i in range(3)] for z in y])

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
        Dense(3, activation='softmax')
    ])

    # model = CovidTweets('datas/metadata.tsv', 'datas/vectors.tsv')
    model.compile(loss='categorical_crossentropy',
             optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, batch_size=128, epochs=10, validation_split=0.2)
    model.save("models/model.h5")
