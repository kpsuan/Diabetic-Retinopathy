import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model once at the start to avoid reloading it every time
new_model = tf.keras.models.load_model("64x3-CNN.model")

def predict_class(img):
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    st.image(RGBImg, caption='Input Image', use_column_width=True)
    image = np.array(RGBImg) / 255.0
    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)
    if per == 1:
        return 'Diabetic Retinopathy NOT DETECTED'
    else:
        return 'Diabetic Retinopathy DETECTED'

# Streamlit app
st.title("Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Predict the class
    result = predict_class(img)
    
    # Display the result
    st.write(f"Prediction: {result}")
