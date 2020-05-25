import tensorflow as tf
from tensorflow.keras import layers, Model


class BottleneckResidual(Model):
    """ResNet の Bottleneck Block です.
    1 層目の 1x1 conv で ch 次元を削減することで
    2 層目の 3x3 conv の計算量を減らし
    3 層目の 1x1 conv で ch 出力の次元を復元します.

    計算量の多い 2 層目 3x3 conv の次元を小することから bottleneck と呼ばれます.
    """

    def __init__(self, in_ch, out_ch, strides=1, *args, **kwargs):
        """
        Args:
            in_ch(int): input filters
            out_ch(int): output filters
            strides(int): window stride
        """
        super().__init__(*args, **kwargs)

        self.projection = in_ch != out_ch
        inter_ch = out_ch // 4
        params = {
            "padding": "same",
            "kernel_initializer": "he_normal",
            "use_bias": True,
        }

        self.common_layers = [
            layers.Conv2D(inter_ch, kernel_size=1, strides=strides, **params),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(inter_ch, kernel_size=3, strides=1, **params),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_ch, kernel_size=1, strides=1, **params),
            layers.BatchNormalization(),
        ]

        if self.projection:
            self.projection_layers = [
                layers.Conv2D(out_ch, kernel_size=1, strides=strides, **params),
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
    """ResNet50 です.
    要素は
    conv * 1
    resblock(conv * 3) * 3
    resblock(conv * 3) * 4
    resblock(conv * 3) * 6
    resblock(conv * 3) * 3
    dense * 1
    から構成されていて, conv * 49 + dense * 1 の 50 層です.
    """

    def __init__(self, output_size=1000, *args, **kwargs):
        """
        Args:
            output_size(int): num of class
        """
        super().__init__(*args, **kwargs)

        self.layers_ = [
            layers.Conv2D(64, 7, 2, padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            BottleneckResidual(64, 256),
            BottleneckResidual(256, 256),
            BottleneckResidual(256, 256),
            BottleneckResidual(256, 512, 2),
            BottleneckResidual(512, 512),
            BottleneckResidual(512, 512),
            BottleneckResidual(512, 512),
            BottleneckResidual(512, 1024, 2),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 2048, 2),
            BottleneckResidual(2048, 2048),
            BottleneckResidual(2048, 2048),
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                output_size, activation="softmax", kernel_initializer="he_normal"
            ),
        ]

    def call(self, inputs):
        """
        Args:
            inputs(tf.Tensor):
        Returns:
            inputs(tf.Tensor):
        """
        for layer in self.layers_:
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


def functional_bottleneck_residual(x, in_ch, out_ch, strides=1):
    params = {
        "padding": "same",
        "kernel_initializer": "he_normal",
        "use_bias": True,
    }
    inter_ch = out_ch // 4
    h1 = layers.Conv2D(inter_ch, kernel_size=1, strides=strides, **params)(x)
    h1 = layers.BatchNormalization()(h1)
    h1 = layers.ReLU()(h1)
    h1 = layers.Conv2D(inter_ch, kernel_size=3, strides=1, **params)(h1)
    h1 = layers.BatchNormalization()(h1)
    h1 = layers.ReLU()(h1)
    h1 = layers.Conv2D(out_ch, kernel_size=1, strides=1, **params)(h1)
    h1 = layers.BatchNormalization()(h1)

    if in_ch != out_ch:
        h2 = layers.Conv2D(out_ch, kernel_size=1, strides=strides, **params)(x)
        h2 = layers.BatchNormalization()(h2)
    else:
        h2 = x

    h = layers.Add()([h1, h2])
    h = layers.ReLU()(h)
    return h


def functional_resnet50(input_shape, output_size):
    inputs = layers.Input(batch_input_shape=input_shape)
    x = layers.Conv2D(64, 7, 2, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = functional_bottleneck_residual(x, 64, 256)
    x = functional_bottleneck_residual(x, 256, 256)
    x = functional_bottleneck_residual(x, 256, 256)

    x = functional_bottleneck_residual(x, 256, 512, 2)
    x = functional_bottleneck_residual(x, 512, 512)
    x = functional_bottleneck_residual(x, 512, 512)
    x = functional_bottleneck_residual(x, 512, 512)

    x = functional_bottleneck_residual(x, 512, 1024, 2)
    x = functional_bottleneck_residual(x, 1024, 1024)
    x = functional_bottleneck_residual(x, 1024, 1024)
    x = functional_bottleneck_residual(x, 1024, 1024)
    x = functional_bottleneck_residual(x, 1024, 1024)
    x = functional_bottleneck_residual(x, 1024, 1024)

    x = functional_bottleneck_residual(x, 1024, 2048, 2)
    x = functional_bottleneck_residual(x, 2048, 2048)
    x = functional_bottleneck_residual(x, 2048, 2048)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        output_size, activation="softmax", kernel_initializer="he_normal"
    )(x)
    return Model(inputs=inputs, outputs=outputs)


class ResNet101(ResNet50):
    """ResNet101 です.
    要素は
    conv * 1
    resblock(conv * 3) * 3
    resblock(conv * 3) * 4
    resblock(conv * 3) * 23
    resblock(conv * 3) * 3
    dense * 1
    から構成されていて, conv * 101 + dense * 1 の 101 層です.
    """

    def __init__(self, output_size=1000, *args, **kwargs):
        """
        Args:
            output_size(int): num of class
        """
        super().__init__(*args, **kwargs)

        self.layers_ = [
            layers.Conv2D(64, 7, 2, padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]
        out_ch = 256
        for i in range(3):
            in_ch = out_ch // 4 if i == 0 else out_ch
            self.layers_.append(BottleneckResidual(in_ch, out_ch))

        out_ch = 512
        for i in range(4):
            in_ch = out_ch // 4 if i == 0 else out_ch
            strides = 2 if i == 0 else 1
            self.layers_.append(BottleneckResidual(in_ch, out_ch, strides))

        out_ch = 1024
        for i in range(23):
            in_ch = out_ch // 4 if i == 0 else out_ch
            strides = 2 if i == 0 else 1
            self.layers_.append(BottleneckResidual(in_ch, out_ch, strides))

        out_ch = 2048
        for i in range(3):
            in_ch = out_ch // 4 if i == 0 else out_ch
            strides = 2 if i == 0 else 1
            self.layers_.append(BottleneckResidual(in_ch, out_ch, strides))

        self.layers_ += [
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                output_size, activation="softmax", kernel_initializer="he_normal"
            ),
        ]
