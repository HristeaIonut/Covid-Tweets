import csv
import pickle as pkl

result = list()
with open("corona_tweets_30.csv", encoding='UTF-8') as csv_file:
    content = csv.reader(csv_file)
    i = 0
    for row in content:
        if i == 0:
            i += 1
            continue
        result.append((row[0], row[1]))

pkl.dump(result, open("data.pkl", "wb"))