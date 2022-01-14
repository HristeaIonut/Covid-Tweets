import pickle as pkl
import tweepy
import os

DATASET_SIZE = 40000
CONSUMER_KEY = os.environ.get('CONSUMER_KEY')
CONSUMER_SECRET = os.environ.get('CONSUMER_SECRET')

result = pkl.load(open("../datas/data.pkl", "rb"))
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
results = list()
count = -1
for i in result:
    if count == DATASET_SIZE:
        break
    count += 1

    print(count)
    try:
        tweet = api.get_status(id=int(i[0]), tweet_mode='extended')
        if tweet.retweeted_status:
            results.append((tweet.retweeted_status.full_text, i[1]))
        else:
            results.append((tweet.full_text, i[1]))
    except:
        pass

f = open("../datas/data_with_retweet.pkl", "wb")
pkl.dump(results, f)
