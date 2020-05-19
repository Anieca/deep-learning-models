import tensorflow as tf
from tensorflow.keras import layers, Model


class BlockCBR(Model):
    """Conv-BatchNorm-Relu Block."""

    def __init__(self, filters, kernel_size, strides):
        """
        Args:
            filters(int): num of filters
            kernel_size(int): filter shape
            strides(int): window stride
        """
        super().__init__()

        params = {
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": "same",
            "use_bias": True,
            "kernel_initializer": "he_normal",
        }

        self._layers = [
            layers.Conv2D(**params),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]

    def call(self, inputs):
        """
        Args:
            inputs(tf.Tensor):
        Returns:
            inputs(tf.Tensor):
        """
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

    def build_graph(self, input_shape):
        """
        Args:
            input_shape(tuple): (batchsize, height, width, channel)
        """
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        self.call(inputs)


class VGG16(Model):
    def __init__(self, output_size=1000):
        """
        Args:
            output_size(int): num of class
        """
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
            BlockCBR(256, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            BlockCBR(512, 3, 1),
            BlockCBR(512, 3, 1),
            BlockCBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            BlockCBR(512, 3, 1),
            BlockCBR(512, 3, 1),
            BlockCBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            layers.Flatten(),
            layers.Dense(4096),
            layers.Dense(4096),
            layers.Dense(output_size, activation="softmax"),
        ]

    def call(self, inputs):
        """
        Args:
            inputs(tf.Tensor):
        Returns:
            inputs(tf.Tensor):
        """
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

    def build_graph(self, input_shape):
        """
        Args:
            input_shape(tuple): (batchsize, height, width, channel)
        """
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        self.call(inputs)
