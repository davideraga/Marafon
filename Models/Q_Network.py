import tensorflow as tf
from tensorflow import keras


class QNet(tf.keras.Model):

    def __init__(self,  input_shape=(780,), num_actions=16):
        super().__init__()
        self.layer1 = keras.layers.Dense(512, activation="relu", input_shape=input_shape)
        #self.layer2 = keras.layers.Dense(128, activation="relu")
        self.layer3 = keras.layers.Dense(num_actions)

    #@tf.function
    def call(self, inputs):
        x1 = self.layer1(inputs)
        #x2 = self.layer2(x1)
        return self.layer3(x1)
