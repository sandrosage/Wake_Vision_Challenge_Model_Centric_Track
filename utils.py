import tensorflow as tf
from typing import Tuple

def get_preprocessing(input_shape: Tuple[int, int, int]):
    return tf.keras.Sequential([tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

def get_prep_augmentation(input_shape: Tuple[int, int, int]):
    return tf.keras.Sequential([
        get_preprocessing(input_shape=input_shape),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2)
        ])

def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2)
        ])