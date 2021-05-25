import tensorflow as tf

''' MNIST 데이터 설정 '''
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

''' Normalization (0~255 사이의 값이므로) '''
x_train = x_train / 255.0
x_test = x_test / 255.0

''' 원-핫 인코딩 (어떤 경우에는 to_categorical()을 쓰기도 한다) '''
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10) 

''' FNN 설정 '''
model3 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # (60000, 28, 28) : 6만장의 28*28 크기의 데이터
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

''' 모델 빌드 (예시) '''
def lr_schedule(epoch):     # 기본 learning rate 값은 0.1
  if epoch < 10:
    lr = 0.1 + (epoch*0.1)  # 첫 10 epoch 동안은 기본 lr 값에 epoch*0.1 만큼을 추가로 더함
  else:
    lr = 1.0 - (epoch*0.04) # 이후 epoch 동안 lr 은 1.0 에서 epoch*0.04 만큼을 줄여나감
  return lr

callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

model3.compile(
    optimizer=tf.keras.optimizers.SGD(), 
    loss=tf.keras.losses.CategoricalCrossentropy(),    # loss='categorical_crossentropy' 대신에 이렇게 적어도 됨
    metrics=['accuracy'])

history = model3.fit(
  x_train, 
  y_train, 
  epochs=25, 
  validation_data=(x_test, y_test), 
  callbacks=[callback], 
  verbose=1
)
