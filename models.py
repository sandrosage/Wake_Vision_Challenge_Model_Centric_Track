from tensorflow_model_optimization.python.core.keras.compat import keras
from typing import Tuple

def get_baseline_model(input_shape: Tuple[int, int, int], name: str):
    inputs = keras.Input(shape=input_shape)
    #
    x = keras.layers.Conv2D(8, (3,3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(16, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(24, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(30, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(34, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(37, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.GlobalAveragePooling2D()(x)
    #
    x = keras.layers.Dense(37)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    outputs = keras.layers.Dense(2)(x)

    return keras.Model(inputs, outputs, name=name)