import pickle as pkl
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import tensorflow.ragged as ragged
from keras.layers import Input, Embedding, LSTM, Dense, Activation
from tensorflow.python.ops.ragged.ragged_tensor import Ragged
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
X = np.asarray(X)
y = np.asarray(y)
X = letters_to_numbers(X)
