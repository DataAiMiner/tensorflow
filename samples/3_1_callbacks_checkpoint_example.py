mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train, x_valid = x_train/255.0, x_valid/255.0

tf.keras.backend.set_floatx('float64')

model = tf.keras.models.Sequential([
  # Flatten으로 shape 펼치기
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # Dense Layer
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  # Classification을 위한 Softmax
  tf.keras.layers. Dense(10, activation='softmax'),
])
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

checkpoint_path = "my_checkpoint.ckpt"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_path,
  save_weights_only=True,
  save_best_only=True,
  monitor='val_loss',
  verbose=1)

history = model.fit(x_train, 
                    y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=30,
                    callbacks=[checkpoint])

model.load_weights(checkpoint_path)

model.evaluate(x_valid, y_valid)

return model
