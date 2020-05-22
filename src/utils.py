import argparse
import tensorflow as tf
from datetime import datetime

from src.vgg16 import VGG16
from src.resnet50 import ResNet50


def get_current_time():
    """現在時刻を取得します.
    Returns:
        (str):
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(name):
    """データセットを読み込みます.
    Args:
        name(str):
    Returns:
        (tuple): (np.ndarray, np.ndarray)
        (tuple): (np.ndarray, np.ndarray)
    """
    data = eval(f"tf.keras.datasets.{name.lower().replace('-', '_')}")
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    if train_images.ndim == 3:
        train_images = train_images[:, :, :, tf.newaxis]
        test_images = test_images[:, :, :, tf.newaxis]

    train_labels = train_labels.astype("float32")
    test_labels = test_labels.astype("float32")
    return (train_images, train_labels), (test_images, test_labels)


def load_model(name, output_size):
    """モデルを読み込みます.
    Args:
        name(str):
        output_size(int):
    Returns:
        model(Object): tf.keras.Model を継承したオブジェクト
    """
    name = name.lower()
    if name == "vgg16":
        model = VGG16(output_size)
    elif name == "resnet50":
        model = ResNet50(output_size)
    else:
        raise KeyError
    return model


def get_args():
    """引数をパースして取得します.
    Returns:
        args(argparse.Namespace)
    """
    models = ["vgg16", "resnet50"]
    datas = ["mnist", "fashion-mnist", "cifar10", "cifar100"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", "-a", choices=models, default="VGG16")
    parser.add_argument("--data", "-d", choices=datas, default="mnist")
    parser.add_argument("--logdir", "-l", default="./logs")
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--max-epoch", "-e", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    return parser.parse_args()