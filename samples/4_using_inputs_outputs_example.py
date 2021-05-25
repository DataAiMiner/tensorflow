import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

tf.random.set_seed(2020)

''' CIFAR-10 dataset setting'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
        return dict

train1 = unpickle('./data_batch_1')
x_train = train1[b'data'].reshape(10000,32,32,3)
y_train = train1[b'labels']
y_train = np.array(y_train, dtype = 'uint8')
y_train = np.expand_dims(y_train, axis=1)
x_train = x_train / 255.0

inputs = tf.keras.Input(shape=(32, 32, 3))
x = inputs
x = layers.Conv2D(16, 5, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(16, 5, activation='relu', padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16)(x)
x = layers.Dense(10, activation='softmax')(x)

outputs = x
model4_1 = tf.keras.Model(inputs, outputs)
model4_1.summary()

''' 모델 전체의 파라미터 수를 35000개 이하로 '''
inputs = tf.keras.Input(shape=(32, 32, 3))
x2 = inputs
x2 = layers.Conv2D(32, 3, activation='relu', padding='valid')(x2)
x2 = layers.MaxPooling2D(2,2)(x2)
x2 = layers.Conv2D(12, 3, activation='relu', padding='valid')(x2)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(15)(x2)
x2 = layers.Dense(10, activation='softmax')(x2)

outputs = x2
model4_2 = tf.keras.Model(inputs, outputs)
model4_2.summary()  

model4_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model4_2.fit(x_train[:5000], y_train[:5000], epochs=5)
