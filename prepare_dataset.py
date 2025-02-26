from datasets import load_dataset, load_dataset_builder, get_dataset_split_names
import tensorflow as tf #for designing and training the model 
import pandas as pd
import pyarrow
import os
from PIL import Image
import numpy as np
from typing import Literal

arrow_dir = "datasets/Harvard-Edge___wake-vision/default/0.0.0/19c27e44926386c2dba2561cf71356b71d81a38b/"
arrow_file = "wake-vision-train_quality-00649-of-00667.arrow"
parquet_dir = "hub/datasets--Harvard-Edge--Wake-Vision/snapshots/19c27e44926386c2dba2561cf71356b71d81a38b/data/"
parquet_file = "train_quality_part_0-00003-of-00030.parquet"

#some hyperparameters 
#Play with them!
input_shape = (80,80,3) # changed it from resolution 50 -> 80

#load dataset
ds_builder = load_dataset_builder("Harvard-Edge/Wake-Vision")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(get_dataset_split_names("Harvard-Edge/Wake-Vision"))
ds = load_dataset("Harvard-Edge/Wake-Vision")

df = pd.read_parquet(os.path.join(parquet_dir,parquet_file))
print("-------------------------")
print(df.loc[0])

# Replace 'your_file.arrow' with the path to your .arrow file
# table = pyarrow.ipc.open_file("datasets/Harvard-Edge___wake-vision/default/0.0.0/19c27e44926386c2dba2561cf71356b71d81a38b/wake-vision-train_quality-00661-of-00667.arrow").read_all()
with pyarrow.memory_map(os.path.join(arrow_dir, arrow_file), 'r') as source:
         table = pyarrow.ipc.open_stream(source).read_all()
# Convert to a Pandas DataFrame
df = table.to_pandas()

# Display the DataFrame
# print(df.loc[0]["image"]["bytes"])
    
train_ds = ds['train_quality'].to_tf_dataset(columns='image', label_cols='person')
val_ds = ds['validation'].to_tf_dataset(columns='image', label_cols='person')
test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

# Create directories to store images and labels
def create_dataset(ds_train, ds_val, ds_test, dirname: str = "wake_vision_dataset"):
        os.makedirs(dirname, exist_ok=True)
        dir_list = ["test", "val", "train"]
        for dir in dir_list:
            os.makedirs(os.path.join(dirname, dir), exist_ok=True)
            os.makedirs(os.path.join(dirname, dir,'images'), exist_ok=True)
            os.makedirs(os.path.join(dirname, dir,'labels'), exist_ok=True)
            # Iterate through the dataset and save images and labels
            if dir == "train":
                dataset = ds_train
            elif dir == "val":
                  dataset = ds_val
            else:
                  dataset = ds_test
            
            print("Length of ds: ", dataset.cardinality().numpy())
            dataset = dataset.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            print("Image shape: ", next(iter(dataset.take(1)))[0].shape)
            label_list = []
            # Iterate through the dataset and save images and labels
            for i, (image, label) in enumerate(dataset):
                # Convert the image tensor to a NumPy array
                img_array = image.numpy().astype('uint8')  # Convert to uint8 for image saving
                img = Image.fromarray(img_array)  # Create a PIL image from the array
                
                # Save image
                img.save(os.path.join(dirname, dir,'images',f'image_{i}.png'))
                
                # Save label
                label_list.append(label.numpy())

            np.save(os.path.join(dirname, dir, "labels", 'labels.npy'), np.array(label_list))

def create_tf_dataset(dataset, flag: Literal["train", "val", "test"] = "test", dirname: str = "wake_vision_tf"):
    print(f"PROCESS {flag} ----------------------")
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(os.path.join(dirname, flag), exist_ok=True)
    os.makedirs(os.path.join(dirname, flag, str(0)), exist_ok=True)
    os.makedirs(os.path.join(dirname, flag, str(1)), exist_ok=True)
    if flag == "test":
        os.makedirs(os.path.join(dirname, flag, str(-1)), exist_ok=True)
            
    print("Length of ds: ", dataset.cardinality().numpy())
    dataset = dataset.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    print("Image shape: ", next(iter(dataset.take(1)))[0].shape)
    # Iterate through the dataset and save images and labels
    for i, (image, label) in enumerate(dataset):
        # Convert the image tensor to a NumPy array
        img_array = image.numpy().astype('uint8')  # Convert to uint8 for image saving
        img = Image.fromarray(img_array)  # Create a PIL image from the array
        
        # Save image
        img.save(os.path.join(dirname, flag ,str(label.numpy()),f'image_{i}.png'))
      
    print("DONE----------------------------")
# create_dataset(train_ds, val_ds, test_ds)
create_tf_dataset(test_ds, "test")
create_tf_dataset(val_ds, "val")
create_tf_dataset(train_ds, "train")