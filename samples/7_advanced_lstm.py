import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/데이터명.json'
urllib.request.urlretrieve(data_url, '데이터명.json')

vocab_size = 1000      # 단어사전 수
embedding_dim = 16     # 임베딩 차원
sentences = []
labels = []

with open('데이터명.json') as f:
  full_data = json.load(f)
for each_data in full_data:
  sentences.append(each_data['문장을 저장한 key 명칭'])
  labels.append(each_data['레이블을 저장한 key 명칭'])
    
    
train_sentences = sentences[:20000]
train_labels = labels[:20000]

validation_sentences = sentences[20000:]
validation_labels = labels[20000:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token='[OOV]')
tokenizer.fit_on_texts(train_sentences) 

train_sequences = tokenizer.texts_to_sequences(train_sentences) 
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

train_pad = pad_sequences(train_sequences, maxlen=120, truncating='post', padding='post')
validation_pad = pad_sequences(validation_sequences, maxlen=120, truncating='post', padding='post')

train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(train_pad, train_labels,
          validation_data=(validation_pad, validation_labels),
          epochs=20)
