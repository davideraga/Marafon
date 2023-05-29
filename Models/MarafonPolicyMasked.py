import tensorflow as tf
from tensorflow import keras


class PolicyNet(tf.keras.Model):

    def __init__(self, num_actions=16, input_shape=(780,)):
        super().__init__()
        self.layer1 = keras.layers.Dense(512, activation="relu", input_shape=input_shape)
        #self.layer2 = keras.layers.Dense(256, activation="relu")
        #self.layer3 = keras.layers.Dense(128, activation="relu")
        self.layer4 = keras.layers.Dense(num_actions)
        self.layer5 = keras.layers.Softmax()

    #@tf.function
    def call(self, inputs, illegal_action_mask):
        x1 = self.layer1(inputs)
        #x2 = self.layer2(x1)
        #x3 = self.layer3(x2)
        x4 = self.layer4(x1)
        x5 = x4 + illegal_action_mask
        return self.layer5(x5)
