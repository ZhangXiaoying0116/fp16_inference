import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH'] ="./amp_log2/"

inputs = keras.Input(shape=(784,), name='digits')
num_units = 4096
# test7: white->gray->white->black
x = layers.Dense(10, name='dense_logits',use_bias=False)(inputs) # white
x = layers.LeakyReLU()(x) # gray
x = layers.Dense(10, name='dense_logits3',use_bias=False)(x) # white
x = tf.pow(x,2) # black
x = layers.Dense(10, name='dense_logits2',use_bias=False)(x)
outputs = layers.Activation('softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
initial_weights = model.get_weights()

# for training
history = model.fit(x_train, y_train,
                    batch_size=8192,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

## for inference
test_score = model.predict(x_test)
print("test_score:",test_score)

