from utils import ModelInspector
import tensorflow as tf

model = tf.keras.models.load_model("models/tf/sepconv.tf")
inspector = ModelInspector(model)
inspector.summary()