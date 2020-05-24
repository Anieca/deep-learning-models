import tensorflow as tf
from tensorflow.keras import layers, Model


class CBR(Model):
    """Convolution, Batch Normalization, ReLU の Block です.
    """

    def __init__(self, filters, kernel_size, strides, *args, **kwargs):
        """
        Args:
            filters(int): num of filters
            kernel_size(int): filter shape
            strides(int): window stride
        """
        super().__init__(*args, **kwargs)

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
    """VGG16 です.
    """

    def __init__(self, output_size=1000, *args, **kwargs):
        """
        Args:
            output_size(int): num of class
        """
        super().__init__(*args, **kwargs)
        self._layers = [
            CBR(64, 3, 1),
            CBR(64, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(128, 3, 1),
            CBR(128, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(256, 3, 1),
            CBR(256, 3, 1),
            CBR(256, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
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


def sequential_vgg16(input_shape, output_size):
    params = {
        "padding": "same",
        "use_bias": True,
        "kernel_initializer": "he_normal",
    }
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, 1, **params, batch_input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2, padding="same"))
    model.add(layers.Conv2D(128, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(128, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2, padding="same"))
    model.add(layers.Conv2D(256, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(256, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(256, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2, padding="same"))
    model.add(layers.Conv2D(512, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(512, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(512, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2, padding="same"))
    model.add(layers.Conv2D(512, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(512, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(512, 3, 1, **params))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2, padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096))
    model.add(layers.Dense(4096))
    model.add(layers.Dense(output_size, activation="softmax"))
    return model


def functional_cbr(x, filters, kernel_size, strides):
    params = {
        "filters": filters,
        "kernel_size": kernel_size,
        "strides": strides,
        "padding": "same",
        "use_bias": True,
        "kernel_initializer": "he_normal",
    }

    x = layers.Conv2D(**params)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def functional_vgg16(input_shape, output_size):
    inputs = layers.Input(batch_input_shape=input_shape)
    x = functional_cbr(inputs, 64, 3, 1)
    x = functional_cbr(x, 64, 3, 1)
    x = layers.MaxPool2D(2, padding="same").call(x)
    x = functional_cbr(x, 128, 3, 1)
    x = functional_cbr(x, 128, 3, 1)
    x = layers.MaxPool2D(2, padding="same").__call__(x)
    x = functional_cbr(x, 256, 3, 1)
    x = functional_cbr(x, 256, 3, 1)
    x = functional_cbr(x, 256, 3, 1)
    x = layers.MaxPool2D(2, padding="same")(x)
    x = functional_cbr(x, 512, 3, 1)
    x = functional_cbr(x, 512, 3, 1)
    x = functional_cbr(x, 512, 3, 1)
    x = layers.MaxPool2D(2, padding="same")(x)
    x = functional_cbr(x, 512, 3, 1)
    x = functional_cbr(x, 512, 3, 1)
    x = functional_cbr(x, 512, 3, 1)
    x = layers.MaxPool2D(2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.Dense(4096)(x)
    outputs = layers.Dense(output_size, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)
