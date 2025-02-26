from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
from datasets import load_dataset #for downloading the Wake Vision Dataset
import tensorflow as tf #for designing and training the model 

model_name = 'wv_k_8_c_5'

#some hyperparameters 
#Play with them!
input_shape = (80,80,3)
batch_size = 512
learning_rate = 0.001
epochs = 100

#model architecture (with Quantization Aware Training - QAT)
#Play with it!
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

model = keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#load dataset
ds = load_dataset("Harvard-Edge/Wake-Vision")
    
train_ds = ds['train_quality'].to_tf_dataset(columns='image', label_cols='person')
val_ds = ds['validation'].to_tf_dataset(columns='image', label_cols='person')
test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

data_augmentation = tf.keras.Sequential([
    data_preprocessing,
    #add some data augmentation 
    #Play with it!
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])
    
train_ds = train_ds.shuffle(1000).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)

#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_sparse_categorical_accuracy',
    mode='max', save_best_only=True)
    
#training
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[model_checkpoint_callback])

#Post Training Quantization (PTQ)
model = tf.keras.models.load_model(model_name + ".tf")