import tensorflow as tf
from src.resnet50 import BottleneckResidual, ResNet50, ResNet101
from src.resnet50 import functional_resnet50


def test_bottlenect_res_block_not_projection():
    input_shape = (10, 224, 224, 256)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_shape = (10, 224, 224, 256)

    model = BottleneckResidual(256, 256)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()


def test_bottlenect_res_block_projection():
    input_shape = (10, 224, 224, 64)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_shape = (10, 224, 224, 256)

    model = BottleneckResidual(64, 256)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()


def test_resnet50():
    input_shape = (10, 224, 224, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_size = 1000
    output_shape = (10, output_size)

    model = ResNet50(output_size)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()


def test_functional_resnet50():
    input_shape = (10, 224, 224, 3)
    inputs = tf.keras.layers.Input(batch_input_shape=input_shape)
    output_size = 1000
    output_shape = (10, output_size)
    model = functional_resnet50(input_shape, output_size)
    model.summary()
    assert model(inputs).shape == output_shape


def test_resnet101():
    input_shape = (10, 224, 224, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_size = 1000
    output_shape = (10, output_size)

    model = ResNet101(output_size)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()
