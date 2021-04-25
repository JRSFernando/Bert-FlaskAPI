import json
import time
import flask
from flask import Flask, request
from flask_cors import CORS

from bert_initializer import classify_tweets, classify_tweets_bert_base
from tweepy_executor import get_related_tweets
from pre_processor import process

app = Flask(__name__)
CORS(app)

evaluation_tweets = [
    "Cost of Taxpayers for Presidential Golf 74 Days into Presidency. Bush 41: $0 Obama:  $0 Trump: $578,640 Biden: $0",
    "Remember when Donald Trump fired up a crowd of white supremacists and conspiracy theorists to attack the US "
    "Congress? It was three months ago today. Some people seem to have forgotten. Don’t let that happen.",
    "Human rights was always an American Cold War weapon against countries that viewed tangible things like housing, "
    "jobs, and healthcare as actual rights. Read Joseph Massad’s Islam in Liberalism.",
    "Leftist shareholders want to hold accountable/gain power over corporations that have used liberalism as a mask for"
    " limitless greed. The American ruling class is a joke just waiting for a more serious version of Trump to come "
    "along and knock it about.",
    "How did American Christianity get to the point where the mere mention of diversity and representation is "
    "met with accusations of woke liberalism?",
    "American and British liberalism are political philosophies primarily dedicated to never ever understating "
    "anything beyond a surface level.",
    "There are a lot of folks on the American left who are *horrified* by the excesses of cancel culture, "
    "wokeism, etc. Those folks are liberal, too. I disagree with them on many things, but I can live with them. "
    "Which is the whole point of liberalism in the first place: tolerance.",
    "Psychoanalysis only applies to bad people, i.e. people who mispronounce the shibboleths of American "
    "liberalism which I, personally, adore. Like, if you think you're an island of good people in an ocean of bad, "
    "that's probably b/c it's true. Mirrors don't lie! -Lacan, I guess.",
    "America will never knowingly adopt Socialism. But, under the name of liberalism, they will adopt every "
    "fragment of the socialist program, until one day American will be a socialist nation, "
    "without knowing how it happened. - Norman Thomas",
    "Every day, they show us how different our systems and values are and the incompatibility of the two; one "
    "defined by democracy and the other by autocracy. The only solution is to part ways from "
    "this uncompleted, deceitful, and unholy partnership. Independence or Resistance forever"
]


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


@app.route('/evaluate')
def evaluate():
    evaluation = dict()
    tweets = []
    pred_sentences = evaluation_tweets
    process(pred_sentences)
    labels = ['Republican', 'Democratic']
    start = time.time()
    label = classify_tweets(pred_sentences)
    end = time.time()
    print(end - start)
    label_list = []
    for i in range(len(pred_sentences)):
        print(pred_sentences[i], ": \n", labels[label[i]])
        label_list.append(labels[label[i]])
        tweets.append({
            'tweet': pred_sentences[i],
            'classification': labels[label[i]]
        })

    evaluation['tweets'] = tweets
    result = json.dumps(evaluation)
    parsed_tweets = json.loads(result)
    parsed_tweets = flask.jsonify(parsed_tweets)

    return parsed_tweets


if __name__ == '__main__':
    app.run()
