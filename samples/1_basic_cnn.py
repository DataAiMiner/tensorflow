from tensorflow.keras.layers import MaxPool2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

model1 = Sequential([
    Conv2D(10, 5, activation='relu', input_shape=(28, 28, 1)),  # 첫 층에는 꼭 input_shape를 써 주어야 함, 10 자리에는 피쳐맵 수, 5 자리에는 (이전층 피쳐맵 크기)-(이번층 피쳐맵크기)+1 가 들어감
    MaxPool2D(pool_size=(2, 2)),                                # pool_size에 들어가는 수는 이전의 Conv2D 층의 피쳐맵 크기를 이 MaxPool2D의 이미지 피쳐맵 크기로 나눈 것
    Conv2D(20, 5, activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])
model1.summary()
