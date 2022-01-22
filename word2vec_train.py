import io
import pickle

from tensorflow.keras import layers

from models.word2vec import Word2Vec
from utils.skipgram_generator import generate_training_data
import tensorflow as tf
import pickle as pkl
import numpy as np
import string
import re


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


if __name__ == '__main__':
    data = pickle.load(open('datas/data_with_retweet.pkl', 'rb'))
    x, _ = zip(*data)

    vocab_size = 20000
    tweets_ds = tf.data.TextLineDataset(['tweets/tweet_texts.txt'])\
        .filter(lambda foo: tf.cast(tf.strings.length(foo), bool))
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        )
    vectorize_layer.adapt(tweets_ds.batch(1024))
    inverse_vocab = vectorize_layer.get_vocabulary()
    # Vectorize the data in tweets_ds.
    text_vector_ds = tweets_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    print(len(sequences))
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=vocab_size,
        seed=SEED)

    targets = np.array(targets)
    contexts = np.array(contexts)[:,:,0]
    labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)

    embedding_dim = 128
    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    word2vec.fit(dataset, epochs=20)
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()
    out_v = io.open('datas/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('datas/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
