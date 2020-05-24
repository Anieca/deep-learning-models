import tensorflow as tf
from src.vgg16 import CBR, VGG16
from src.vgg16 import sequential_vgg16
from src.vgg16 import functional_vgg16


def test_cbr():

    filters = 64
    kernel_size = 3
    strides = 2
    input_shape = (10, 224, 224, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    model = CBR(filters, kernel_size, strides)

    assert model(inputs).shape == (
        input_shape[0],
        input_shape[1] // strides,
        input_shape[2] // strides,
        filters,
    )

    model.build_graph(input_shape)
    model.summary()


def test_vgg16():
    input_shape = (10, 224, 224, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_size = 1000
    output_shape = (10, output_size)

    model = VGG16(output_size)
    assert model(inputs).shape == output_shape

    model.build_graph(input_shape)
    model.summary()


def test_sequential_vgg16():

    input_shape = (10, 224, 224, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_size = 1000
    output_shape = (10, output_size)

    model = sequential_vgg16(input_shape, output_size)
    model.summary()
    assert model(inputs).shape == output_shape


def test_functional_vgg16():

    input_shape = (10, 224, 224, 3)
    inputs = tf.keras.layers.Input(batch_input_shape=input_shape)
    output_size = 1000
    output_shape = (10, output_size)
    model = functional_vgg16(input_shape, output_size)
    model.summary()
    assert model(inputs).shape == output_shape
