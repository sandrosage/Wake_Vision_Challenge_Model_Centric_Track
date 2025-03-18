
import tensorflow_model_optimization as tfm
from argparse import ArgumentParser
import yaml
import tensorflow as tf
from utils import get_augmentation
from load_dataset import get_single_tf_dataset
import os
import csv
import numpy as np
import tempfile

prune_low_magnitude = tfm.sparsity.keras.prune_low_magnitude
epochs = 2
parser = ArgumentParser()
parser.add_argument(
    "--name",
    required=True,
    type=str,
    help="Name of the model",
    )

args = parser.parse_args()
print("START PRUNING ----------------------")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(f"Representative batch size: {config['ptq_batch_size']}")

input_shape = (config["res"], config["res"], 3)

model_name = args.name
model = tf.keras.models.load_model("models/tf/" + model_name + ".tf")
print(f"MODEL LOADED from {'models/tf/' + model_name + '.tf'}----------------------")

data_augmentation = get_augmentation()
train_ds = get_single_tf_dataset(config["batch_size"], input_shape, config["train_ds_path"])
val_ds = get_single_tf_dataset(config["batch_size"], input_shape, config["val_ds_path"])
test_ds = get_single_tf_dataset(1, input_shape, config["test_ds_path"])
train_ds = train_ds.shuffle(config["shuffle_buffer_size"]).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

end_step = np.ceil(train_ds.cardinality() / config["batch_size"]).astype(np.int32) * epochs
# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfm.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

callbacks = [
  tfm.sparsity.keras.UpdatePruningStep(),
  tfm.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
]
model_for_pruning.summary()
model_for_pruning.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

def representative_dataset():
    for data in train_ds.rebatch(1).take(config["ptq_batch_size"]) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

model_for_export = tfm.sparsity.keras.strip_pruning(model_for_pruning)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open("models/qat/PRU_" + model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter("models/qat/PRU_" + model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")

headers = ['name', 'size_KB', 'size_MB', 'test_acc']

data = [["PRU_" + model_name, (os.path.getsize("models/qat/PRU_" + model_name + ".tflite")/1000), (os.path.getsize("models/qat/PRU_" + model_name + ".tflite")/1000000), (correct/(correct+wrong))]]
# Writing to a CSV file
with open('output.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers only if the file is empty
    if not os.path.isfile('output.csv') or os.stat('output.csv').st_size == 0:
        writer.writerow(headers)  # Write headers

    # Write new data
    writer.writerows(data)