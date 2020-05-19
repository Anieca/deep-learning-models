import tensorflow as tf
from tensorflow.keras import layers, Model


class BlockCBR(Model):
    """Conv-BatchNorm-Relu Block.
    """

    def __init__(self, filters, kernel_size, strides):
        super().__init__()

        params = {
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": "same",
            "use_bias": False,
            "kernel_initializer": "he_normal",
        }

        self.conv = layers.Conv2D(**params)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=False):
        tensor = self.conv(inputs)
        tensor = self.bn(tensor)
        return self.relu(tensor)

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        self.call(inputs)


class VGG16(Model):
    def __init__(self, output_shape):
        super().__init__()
        self._layers = [
            BlockCBR(64, 3, 1),
            BlockCBR(64, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            BlockCBR(128, 3, 1),
            BlockCBR(128, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            BlockCBR(256, 3, 1),
            BlockCBR(256, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            BlockCBR(512, 3, 1),
            BlockCBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            BlockCBR(512, 3, 1),
            BlockCBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            layers.Flatten(),
            layers.Dense(4096),
            layers.Dense(4096),
            layers.Dense(output_shape, activation="softmax"),
        ]

    def call(self, inputs):
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        self.call(inputs)
