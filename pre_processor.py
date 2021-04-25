import re
import nltk
from nltk.corpus import stopwords
import nltk as nlp

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords = stopwords.words('english')
stopwords.append("rt")
stopwords.append("u")
stopwords.append("amp")
stopwords.append("w")
stopwords.append("th")


def process(pred_sentences):
    cleaned_tweets = []
    for tweet in pred_sentences:
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub("[^a-zA-Z]", " ", tweet)
        tweet = tweet.lower()
        tweet = nltk.word_tokenize(tweet)
        tweet = [word for word in tweet if not word in set(stopwords)]
        lemma = nlp.WordNetLemmatizer()
        tweet = [lemma.lemmatize(word) for word in tweet]
        tweet = " ".join(tweet)
        cleaned_tweets.append(tweet)

    return cleaned_tweets
