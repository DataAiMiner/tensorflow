import tensorflow as tf

model_rnn_1 = tf.keras.Sequential([       
    tf.keras.layers.Embedding(1000002, 32),           # Vocabulary의 각 단어는 길이 32의 벡터로 임베딩하여 사용
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(6, activation='softmax')    # 분류하는 클래스의 개수가 6개임
])  

model_rnn_1.summary()


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

''' 모델 개선 '''
model_rnn_2 = tf.keras.Sequential([        
    tf.keras.layers.Embedding(1000002, 32),            # 단어 100만개 + padding 을 위한 '[PAD]' 와 vocabulary 외의 단어를 처리하는 '[OOV]' 2개 = 1000002
    tf.keras.layers.LSTM(50, return_sequences=True),   # LSTM 층을 여러개 사용한다면, 다음 LSTM 층을 위해 return_sequences=True 옵션을 꼭 넣어줘야 한다.
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(6, activation='softmax')
])  

model_rnn_2.summary()

''' 문장 자르기 처리하기 '''
mw_ids = [[123, 124, 346, 76, 66, 1221, 8762, 4574, 66, 1, 7, 999, 2, 4, 98685, 52, 10, 40, 124, 127, 789654, 27, 122, 9, 1991, 4, 8, 15, 16, 23, 42]] 

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
mw_ids_result = pad_sequences(mw_ids, maxlen=25, truncating='post')   # 25단어 이상 들어오면 뒷부분(post)를 자름 / 앞부분을 자르려면 'pre'를 사용
                                                                      # 채우고 싶으면 padding 옵션을 사용

print(mw_ids_result)
