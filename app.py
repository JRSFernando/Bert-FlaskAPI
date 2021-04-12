import json
import time
import flask
from flask import Flask, request
from flask_cors import CORS

from bert_initializer import classify_tweets
from tweepy_executor import get_related_tweets

app = Flask(__name__)
CORS(app)


@app.route('/classifytweets')
def classify():
    search_term = request.args.get('search-term')
    tweets = get_related_tweets(search_term)
    tweet_list = tweets['text'].to_numpy()

    pred_sentences = []
    for i in range(len(tweet_list)):
        pred_sentences.append(tweet_list[i])

    labels = ['Republican', 'Democratic']
    start = time.time()
    label = classify_tweets(pred_sentences)
    end = time.time()
    print(end - start)
    label_list = []
    for i in range(len(pred_sentences)):
        print(pred_sentences[i], ": \n", labels[label[i]])
        label_list.append(labels[label[i]])

    tweets['classification'] = label_list
    result = tweets.to_json(orient="records")
    parsed_tweets = json.loads(result)
    parsed_tweets = flask.jsonify(parsed_tweets)

    return parsed_tweets


if __name__ == '__main__':
    app.run()
