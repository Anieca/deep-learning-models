import tensorflow as tf
from src.resnet50 import BottleneckResBlock, ResNet50


def test_bottlenect_res_block_not_projection():
    input_shape = (10, 224, 224, 256)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_shape = (10, 224, 224, 256)

    model = BottleneckResBlock(256, 256)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()


def test_bottlenect_res_block_projection():
    input_shape = (10, 224, 224, 64)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_shape = (10, 224, 224, 256)

    model = BottleneckResBlock(64, 256)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()


def test_resnet():
    input_shape = (10, 224, 224, 3)
    inputs = tf.zeros((input_shape), dtype=tf.float32)
    output_size = 1000
    output_shape = (10, output_size)

    model = ResNet50(output_size)
    assert model(inputs).shape == output_shape
    model.build_graph(input_shape)
    model.summary()
