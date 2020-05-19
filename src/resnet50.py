import tensorflow as tf
from tensorflow.keras import layers, Model


class BottleneckResBlock(Model):
    """ResNet Bottleneck Block"""

    def __init__(self, in_ch, out_ch, strides=1):
        """
        Args:
            in_ch(int): input filters
            out_ch(int): output filters
            strides(int): window stride
        """
        super().__init__()

        self.projection = in_ch != out_ch
        inter_ch = out_ch // 4

        self.common_layers = [
            layers.Conv2D(inter_ch, kernel_size=1, strides=strides, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(inter_ch, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_ch, kernel_size=1, strides=1, padding="same"),
            layers.BatchNormalization(),
        ]

        if self.projection:
            self.projection_layers = [
                layers.Conv2D(out_ch, kernel_size=1, strides=strides, padding="same"),
                layers.BatchNormalization(),
            ]

        self.concat_layers = [layers.Add(), layers.ReLU()]

    def call(self, inputs):
        """
        Args:
            inputs(tf.Tensor):
        Returns:
            outputs(tf.Tensor):
        """
        h1 = inputs
        h2 = inputs

        for layer in self.common_layers:
            h1 = layer(h1)

        if self.projection:
            for layer in self.projection_layers:
                h2 = layer(h2)

        outputs = [h1, h2]
        for layer in self.concat_layers:
            outputs = layer(outputs)
        return outputs

    def build_graph(self, input_shape):
        """
        Args:
            input_shape(tuple): (batchsize, height, width, channel)
        """
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        self.call(inputs)


class ResNet50(Model):
    """ResNet50"""

    def __init__(self, output_size=1000):
        """
        Args:
            output_size(int): num of class
        """
        super().__init__()

        self._layers = [
            layers.Conv2D(64, 7, 2, padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            BottleneckResBlock(64, 256),
            BottleneckResBlock(256, 256),
            BottleneckResBlock(256, 256),
            BottleneckResBlock(256, 512, 2),
            BottleneckResBlock(512, 512),
            BottleneckResBlock(512, 512),
            BottleneckResBlock(512, 512),
            BottleneckResBlock(512, 1024, 2),
            BottleneckResBlock(1024, 1024),
            BottleneckResBlock(1024, 1024),
            BottleneckResBlock(1024, 1024),
            BottleneckResBlock(1024, 1024),
            BottleneckResBlock(1024, 1024),
            BottleneckResBlock(1024, 2048, 2),
            BottleneckResBlock(2048, 2048),
            BottleneckResBlock(2048, 2048),
            layers.GlobalAveragePooling2D(),
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
