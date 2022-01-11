from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import tweepy
import numpy as np

CONSUMER_KEY = 'Dv7PE2aL7XwwKguIQIjVVdTU1'
CONSUMER_SECRET = '3vBKSLS2z2EeW8HmV83waJfaYTyAYNOi47lBaXScCcna8l8tws'

def convert(max_words, text, max_length):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    tweet = pad_sequences(sequences, maxlen=max_length)
    return tweet

def predict_one(id):

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    data = pkl.load(open("data_with_retweet.pkl", "rb"))
    X, _ = zip(*data)
    one_text = None
    try:
        tweet = api.get_status(id=id, tweet_mode='extended')
        if tweet.retweeted_status:
            one_text = tweet.retweeted_status.full_text
        else:
            one_text = tweet.full_text

        X = np.append(X, one_text)
        X = convert(20000, X, 104)

        model = load_model("model.h5")
        res = model.predict(X)
        if np.round(res[-1]).astype(np.int32) > 0:
            print("Classified \"" + one_text + "\" as positive")
        else:
            print("Classified \"" + one_text + "\" as negative")
    
    except:
        print("Invalid ID")

    


predict_one(1465833001250549769)




