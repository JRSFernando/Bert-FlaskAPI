import tensorflow as tf
from flask import Flask
from transformers import BertTokenizer, TFBertForSequenceClassification
from tweepy_executor import get_related_tweets

app = Flask(__name__)

model = TFBertForSequenceClassification.from_pretrained('./fine-tuned-Bert/')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.route('/')
def classify():
    tweets = get_related_tweets('trump')
    tweetList = tweets['tweet_text'].to_numpy()
    print(tweetList[0])
    pred_sentences = [
        tweetList[0],
        'Gender equality lgbtq should be promoted']
    tf_batch = tokenizer(pred_sentences, max_length=75, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = ['Rep', 'Dem']
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    for i in range(len(pred_sentences)):
        print(pred_sentences[i], ": \n", labels[label[i]])
    return str(tweets)


if __name__ == '__main__':
    app.run()
