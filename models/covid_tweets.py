from tensorflow.keras.layers import TextVectorization, Lambda, GlobalAveragePooling1D, Dropout, Dense
import tensorflow as tf
import numpy as np

class CovidTweets(tf.keras.Model):
    def __init__(self, vocab_file_path, word2vec_file_path):
        super(CovidTweets, self).__init__()
        self.word2vec = tf.convert_to_tensor(np.loadtxt(word2vec_file_path))
        self.model = tf.keras.Sequential([
            TextVectorization(
                vocabulary=vocab_file_path,
                output_mode='int',
                output_sequence_length=60),
            Lambda(lambda x: self.word2vec[x]), #print(tf.convert_to_tensor(self.word2vec[x]))), #[self.word2vec[i] for i in x]),
            GlobalAveragePooling1D(),
            Dense(64),
            Dense(1)
        ])

    def call(self, sentence):
        return self.model(sentence)
