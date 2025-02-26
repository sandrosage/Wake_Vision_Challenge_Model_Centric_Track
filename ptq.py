import tensorflow as tf
import yaml
from load_dataset import get_single_tf_dataset
from utils import get_augmentation

print("START QUANTIZATION ----------------------")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

input_shape = (config["res"], config["res"], 3)

model = tf.keras.models.load_model(config["model_name"] + ".tf")

data_augmentation = get_augmentation()
train_ds = get_single_tf_dataset(config["batch_size"], input_shape, config["train_ds_path"])
test_ds = get_single_tf_dataset(1, input_shape, config["test_ds_path"])
train_ds = train_ds.shuffle(config["shuffle_buffer_size"]).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

def representative_dataset():
    for data in train_ds.rebatch(1).take(config["ptq_batch_size"]) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(config["model_name"] + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter(config["model_name"] + ".tflite")
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