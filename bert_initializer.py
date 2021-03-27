import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained('./fine-tuned-Bert/')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")