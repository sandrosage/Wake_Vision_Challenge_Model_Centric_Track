import tensorflow as tf
from typing import Tuple
import math


class ModelInspector:
    def __init__(self, model):
        self._model = model
        self._model_layers = [layer for layer in self._model.layers]

    def get_maccs_per_layer(self, layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return layer.count_params()
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            _, h_out, w_out, c_out = layer.output_shape
            c_in = layer.input_shape[-1]
            # MACCS = h_out x w_out x c_out x kernel_size x kernel_size x c_in
            return h_out*w_out*c_out*math.prod(layer.kernel_size)*c_in
        if isinstance(layer, tf.keras.layers.SeparableConv2D):
            depth_layer = self.get_maccs_per_layer(tf.keras.layers.DepthwiseConv2D())
        
        else:
            return 0

    def inspect(self):
        params = {}
        for layer in self._model_layers:
            params[layer.name] = {
                'n_params': layer.count_params(),
                'input_shape': layer.input_shape,
                'output_shape': layer.output_shape,
                'activations': math.prod(layer.output_shape[1:]),
                'maccs': self.get_maccs_per_layer(layer)
            }
        return params
    
    def summary(self):
        params = self.inspect()
        print(f"{'Layer':<35}{'MACCs':<20}{'# of Parameters':<20}{'Output_shape':<25}{'Activations':<20}")
        print("=" * 115)
        for name in params:
            output_shape = params[name]['output_shape']
            print(f"{name:<35}{params[name]['maccs']:<20}{params[name]['n_params']:<20}{str(output_shape):<25}{params[name]['activations']:<20}")


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