import tensorflow as tf
from tensorflow import keras


class VNet(tf.keras.Model):

    def __init__(self, input_shape=(780,)):
        super().__init__()
        self.layer1 = keras.layers.Dense(512, activation="relu", input_shape=input_shape)
        # self.layer2 = keras.layers.Dense(128, activation="relu")
        #self.layer3 = keras.layers.Dense(128, activation="relu")
        self.layer4 = keras.layers.Dense(1)

    #@tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        x1 = self.layer1(inputs)
        #x2 = self.layer2(x1)
        #x3 = self.layer3(x2)
        return self.layer4(x1)
