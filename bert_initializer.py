import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained('./fine-tuned-Bert/')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def classify_tweets(pred_sentences):
    tf_batch = tokenizer(pred_sentences, max_length=75, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()

    return label
