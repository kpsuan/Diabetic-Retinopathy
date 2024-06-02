import tensorflow as tf

# Load the Keras model
new_model = tf.keras.models.load_model("64x3-CNN.model")

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
try:
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model saved successfully as model.tflite")
except Exception as e:
    print("Error saving model:", e)
