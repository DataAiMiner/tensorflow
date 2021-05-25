import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

''' 데이터 불러오는 부분은 다 제거함 '''

TRAINING_DIR = 'tmp/train/'
VALIDATION_DIR = 'tmp/validation/'
train_datagen = ImageDataGenerator(
  rescale=1/ 255.0,
  rotation_range=4,
  width_shift_range=0.05,
  height_shift_range=0.05,
  shear_range=0.05,
  zoom_range=0.05,
  horizontal_flip=True,
  fill_mode='nearest',
)

validation_datagen = ImageDataGenerator(rescale=1/ 255.0,)                          # 보통 검증 데이터에는 normalization만 한다

train_generator = train_datagen.flow_from_directory(
  TRAINING_DIR,
  target_size=(300, 300),    # 이미지 크기가 300x300
  batch_size=32,
  class_mode='binary',
)

validation_generator = validation_datagen.flow_from_directory(
  VALIDATION_DIR,
  target_size=(300, 300),
  batch_size=32,
  class_mode='binary',
)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),                                                         # Dense로 넘기기 전에 납작하게 하는거 잊으면 안됨
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) 
# IDG는 자동으로 원-핫 인코딩을 해주므로 만약 마지막 층이 Dense(2이상, activation='softmax') 였다면 loss='categorical_crossentropy' 여야 함

model.fit(train_generator,
          validation_data=(validation_generator),
          epochs=10)

