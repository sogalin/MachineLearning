import numpy as np
import tensorflow as tf
 
def run_model():
    x = np.random.randint(0, 1, size=(512, 224, 224, 3))
    y = np.random.randint(0, 1000, size=512)
    y = tf.keras.utils.to_categorical(y, 1000)
     
    model = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3), weights=None)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam())
     
    res = model.fit(x, y, batch_size=64, epochs=10)

if __name__ == '__main__':
    run_model()
