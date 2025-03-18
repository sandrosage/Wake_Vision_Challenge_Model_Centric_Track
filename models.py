from tensorflow_model_optimization.python.core.keras.compat import keras
from typing import Tuple

def get_model_from_name(name: str):
    if name == "colabnas_k_8":
        return get_baseline_model(input_shape=(80,80,3), name=name)
    
    elif name == "sepconv":
        return get_sepconv_model(input_shape=(80,80,3), name=name)
    
    elif name =="mcunet_vww1":
        return get_mcunet_vww_1(input_shape=(80,80,3), name=name)
    
    elif name == "own_baseline":
        return get_own_baseline_model(input_shape=(80,80,3), name=name)
    
    elif name == "new_baseline":
        return get_new_baseline_model(input_shape=(80,80,3), name=name)
    
    else: 
        print("No valid name, use baseline again")
        return get_baseline_model(input_shape=(80,80,3), name=name)

def get_new_baseline_model(input_shape: Tuple[int, int, int], name: str):
    inputs = keras.Input(shape=input_shape)
    #
    x = keras.layers.SeparableConv2D(8, (3,3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(8, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(16, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(16, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(24, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(32, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(64, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.GlobalAveragePooling2D()(x)
    #
    x = keras.layers.Dense(47)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    outputs = keras.layers.Dense(2)(x)

    return keras.Model(inputs, outputs, name=name)

def get_own_baseline_model(input_shape: Tuple[int, int, int], name: str):
    inputs = keras.Input(shape=input_shape)
    #
    x = keras.layers.SeparableConv2D(8, (3,3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(8, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(16, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(16, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(24, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(32, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(64, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.Conv2D(2, (1,1), padding='valid')(x)
    outputs = keras.layers.Reshape((2,))(x)

    return keras.Model(inputs, outputs, name=name)

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

def get_sepconv_model(input_shape: Tuple[int, int, int], name: str):
    inputs = keras.Input(shape=input_shape)
    #
    x = keras.layers.SeparableConv2D(8, (3,3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(16, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(24, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(32, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(40, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.SeparableConv2D(47, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    x = keras.layers.GlobalAveragePooling2D()(x)
    #
    x = keras.layers.Dense(47)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #
    outputs = keras.layers.Dense(2)(x)

    return keras.Model(inputs, outputs, name=name)

def get_mcunet_vww_1(input_shape: Tuple[int, int, int], name: str):
    inputs = keras.Input(shape=input_shape)
    #
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
    x = keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(8, (1,1), padding='valid')(x)
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    # add
    y = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    # add
    x = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
    x = keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    # add
    y = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(120, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    # add
    x = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
    x = keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
    # add
    x = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(192, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    # add
    x = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = keras.layers.DepthwiseConv2D((5,5),  padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(480, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
    # add
    y = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(384, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
    # add
    x = keras.layers.Add()([x, y])
    #
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(288, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
    x = keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(160, (1,1), padding='valid')(x)
    #
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(2, (1,1), padding='valid')(x)
    outputs = keras.layers.Reshape((2,))(x)

    return keras.Model(inputs, outputs, name=name)