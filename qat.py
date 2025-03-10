
import tensorflow_model_optimization as tfm
from argparse import ArgumentParser
import yaml
import tensorflow as tf
from utils import get_augmentation
from load_dataset import get_single_tf_dataset
import os
import csv

parser = ArgumentParser()
parser.add_argument(
    "--name",
    required=True,
    type=str,
    help="Name of the model",
    )

args = parser.parse_args()
print("START QUANTIZATION ----------------------")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(f"Representative batch size: {config['ptq_batch_size']}")

input_shape = (config["res"], config["res"], 3)

model_name = "models/tf/" + args.name
model = tf.keras.models.load_model(model_name + ".tf")
q_aware_model = tfm.quantization.keras.quantize_model(model)
opt = tf.keras.optimizers.Adam(learning_rate=config["lr"])

q_aware_model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

data_augmentation = get_augmentation()
train_ds = get_single_tf_dataset(config["batch_size"], input_shape, config["train_ds_path"])
val_ds = get_single_tf_dataset(config["batch_size"], input_shape, config["val_ds_path"])
test_ds = get_single_tf_dataset(1, input_shape, config["test_ds_path"])
train_ds = train_ds.shuffle(config["shuffle_buffer_size"]).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
q_aware_model.fit(train_ds, epochs=1, validation_data=val_ds)

def representative_dataset():
    for data in train_ds.rebatch(1).take(config["ptq_batch_size"]) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open("models/qat/QAT_" + model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter("models/qat/QAT_" + model_name + ".tflite")
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

data = [["QAT_" + model_name, (os.path.getsize(model_name + ".tflite")/1000), (os.path.getsize(model_name + ".tflite")/1000000), (correct/(correct+wrong))]]
# Writing to a CSV file
with open('output.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers only if the file is empty
    if not os.path.isfile('output.csv') or os.stat('output.csv').st_size == 0:
        writer.writerow(headers)  # Write headers

    # Write new data
    writer.writerows(data)