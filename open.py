import pickle as pkl
import tweepy
CONSUMER_KEY = 'Dv7PE2aL7XwwKguIQIjVVdTU1'
CONSUMER_SECRET = '3vBKSLS2z2EeW8HmV83waJfaYTyAYNOi47lBaXScCcna8l8tws'

result = pkl.load(open("data.pkl", "rb"))
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
api = tweepy.API(auth)
count = 0
results = list()

for i in result:
    try:
        tweet = api.get_status(id=int(i[0]), tweet_mode='extended')
        if tweet.retweeted_status:
            results.append((tweet.retweeted_status.full_text, i[1]))
        else:
            results.append((tweet.full_text, i[1]))
    except:
        pass

f = open("data_with_retweet.pkl", "wb")
pkl.dump(results, f)