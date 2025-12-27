import tensorflow as tf
from tensorflow import keras
from keras.layers import TFSMLayer

print("TensorFlow version:", tf.__version__)

MODEL_DIR = "./1"
OUTPUT_MODEL = "model.keras"

print("Wrapping SavedModel with TFSMLayer...")

model = keras.Sequential([
    TFSMLayer(
        MODEL_DIR,
        call_endpoint="serving_default"
    )
])

print("Saving wrapped model as .keras ...")
model.save(OUTPUT_MODEL)

print("✅ model.keras created successfully")
