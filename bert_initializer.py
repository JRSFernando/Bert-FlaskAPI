import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('./quantized-bert/')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def classify_tweets(pred_sentences):
    encoding = tokenizer(pred_sentences, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    tensor = outputs.logits.detach().numpy()
    predictions = tf.nn.softmax(tensor, axis=-1)
    label = tf.argmax(predictions, axis=1)
    label = label.numpy()

    return label
