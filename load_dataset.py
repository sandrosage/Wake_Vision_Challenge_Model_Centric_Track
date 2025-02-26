import tensorflow as tf
from typing import Tuple

def get_single_tf_dataset(batch_size: int, input_shape: Tuple[int, int, int], path: str):
    return tf.keras.utils.image_dataset_from_directory(
            directory= path,
            labels='inferred',
            label_mode='categorical',
            color_mode="rgb",
            batch_size=batch_size,
            image_size=input_shape[0:2],
            shuffle=True,
            seed=11
        )

def get_tf_datasets(batch_size: int, input_shape: Tuple[int, int, int], train_path: str, val_path: str, test_path: str):
    train_ds = get_single_tf_dataset(batch_size, input_shape, train_path)

    val_ds = get_single_tf_dataset(batch_size, input_shape, val_path)

    test_ds = get_single_tf_dataset(1, input_shape, test_path)
    
    return train_ds, val_ds, test_ds

