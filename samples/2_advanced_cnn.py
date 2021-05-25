import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 

tf.random.set_seed(100)
 
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x[:10000], test_x[:10000]
train_y, test_y = train_y[:10000], test_y[:10000]

train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

# 원-핫 인코딩 / loss='sparse_categorical_crossentropy'을 쓰려면 원-핫 인코딩 하면 안됨
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

train_normalized = train_x / 255.0
test_normalized = test_x / 255.0

model2 = Sequential()
model2.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_normal', input_shape=(28, 28, 1)))  
# kernel_initializer='glorot_normal' 부분은 Xavier Initialization을 뜻함
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(15, activation='relu', kernel_initializer='glorot_normal'))
model2.add(Dense(10, activation='softmax'))

model2.summary()

sgdm = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
model2.compile(optimizer=sgdm, loss='categorical_crossentropy', metrics=['accuracy'])
# 마지막 출력층이 Dense(1, activation='sigmoid') 라면, 원-핫 인코딩 여부와는 무관하게 loss='binary_crossentropy'
# 마지막 출력층이 Dense(2이상, activation='softmax') 라면, 원-핫 인코딩 했을 경우에는 loss='categorical_crossentropy' 안했을 경우에는 loss='sparse_categorical_crossentropy'


model2.fit(train_normalized, train_y, epochs=5, batch_size=32, verbose=1)

model2.evaluate(test_normalized, test_y, verbose=1) 
