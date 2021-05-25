import tensorflow as tf
from tensorflow.keras import layers 

def subnetwork(x):
    x1 = layers.Conv2D(filters = 5, kernel_size = 3, padding='same')(x)       # 문제에서 filter의 widthXheight가 3x3이라고 알려줬다. 
                                                                              # (엄밀히 말하면 kernel과 filter는 다른데 통상적으로 구분하지 않고 사용(https://data-science-hi.tistory.com/128)
    x2 = layers.Conv2D(filters = 5, kernel_size = 3, padding='same')(x)
    x2 = layers.Conv2D(filters = 5, kernel_size = 3, padding='same')(x2)      # 혼자만 (x2)가 들어가는 것은 가운데 화살표의 Conv가 두개 들어있기 때문.
    
    x3 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)  # pooling size가 3x3인 MaxPooling 연산
    
    return layers.concatenate([x1, x2, x3], axis=3)                           # concatenate를 통해 x1, x2, x3을 병렬적으로 수행할 수 있도록 해준다.


inputs = tf.keras.Input(shape=(32, 32, 3))
x = inputs
x = subnetwork(x)                                           # 서브네트워크 사용
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)      # pool_size=(2,2) : 피쳐맵의 width와 height를 1/2로 줄인다 -> 32에서 16이 됨
x = subnetwork(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)      # pool_size=(2,2) : 피쳐맵의 width와 height를 1/2로 줄인다 -> 16에서 8이 됨
x = subnetwork(x)
x = layers.AveragePooling2D(pool_size=(8,8), strides=1)(x)  # pool_size=(8,8) : 피쳐맵의 width와 height를 1x1로 줄인다 -> (8,8)을 써서 마지막으로 넘겨주는 것을 (1,1)로 만듬
x = layers.Dense(5, activation='softmax')(x)                # 5개 클래스로 분류

outputs = x
model5 = tf.keras.Model(inputs, outputs)
model5.summary()
