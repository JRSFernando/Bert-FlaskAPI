import tensorflow as tf
import torch
import os
from transformers import BertTokenizer, TFBertForSequenceClassification

# model = TFBertForSequenceClassification.from_pretrained('./political-sentiment-bert-base/')
model = torch.jit.load("bert_traced_pruned_quant.pt")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def classify_tweets(pred_sentences):
    encoding = tokenizer(pred_sentences, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids']
    outputs = model(input_ids, attention_mask, token_type_ids)
    tensor = outputs['logits'].detach().numpy()
    predictions = tf.nn.softmax(tensor, axis=-1)
    label = tf.argmax(predictions, axis=1)
    label = label.numpy()

    return label


def classify_tweets_bert_base(pred_sentences):
    tokens = tokenizer(pred_sentences, max_length=75, padding=True, truncation=True, return_tensors='tf')
    outputs = model(tokens)
    predictions = tf.nn.softmax(outputs[0], axis=-1)
    label = tf.argmax(predictions, axis=1)
    label = label.numpy()

    return label


def get_model_size():
    if isinstance(model, torch.jit._script.RecursiveScriptModule):
        size = os.path.getsize("bert_traced_pruned_quant.pt") / 1e6
    else:
        size = os.path.getsize("political-sentiment-bert-base/tf_model.h5") / 1e6

    return size
