import tensorflow as tf 
from models import get_baseline_model
from utils import get_augmentation, get_preprocessing, get_prep_augmentation
import yaml
from load_dataset import get_tf_datasets

print("START MODEL TRAINING ------------------------------")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

input_shape = (config["res"], config["res"], 3)

model = get_baseline_model(input_shape=input_shape, name=config["model_name"])
#compile model
opt = tf.keras.optimizers.Adam(learning_rate=config["lr"])

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

data_augmentation = get_augmentation()
train_ds, val_ds, test_ds = get_tf_datasets(config["batch_size"], input_shape, config["train_ds_path"], config["val_ds_path"], config["test_ds_path"])
train_ds = train_ds.shuffle(config["shuffle_buffer_size"]).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model.name + ".tf",
    monitor='val_categorical_accuracy',
    mode='max', save_best_only=True)

model.fit(train_ds, epochs=config["epochs"], validation_data=val_ds, callbacks=[model_checkpoint_callback])

print("MODEL TRAINING DONE ----------------------------")