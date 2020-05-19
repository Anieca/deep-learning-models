import tensorflow as tf
from src.vgg16 import BlockCBR, VGG16


def test_block_cbr():

    filters = 64
    kernel_size = 3
    strides = 2
    input_shape = (10, 256, 256, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    model = BlockCBR(filters, kernel_size, strides)

    model.build_graph(input_shape)
    assert model(inputs).shape == (
        input_shape[0],
        input_shape[1] // strides,
        input_shape[2] // strides,
        filters,
    )

    model.summary()


def test_vgg16():
    input_shape = (10, 256, 256, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_shape = 10

    model = VGG16(output_shape)
    model.build_graph(input_shape)
    model(inputs).shape == output_shape
    model.summary()
