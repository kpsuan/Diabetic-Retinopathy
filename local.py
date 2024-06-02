import cv2
import numpy as np
from keras.models import load_model

# Set the size of the input images
img_size = (224, 224)

# Load the trained model for diabetic retinopathy
model = load_model('64x3-CNN.model')

# Open the camera stream
cap = cv2.VideoCapture(0)

# Loop through the frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    img = cv2.resize(frame, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Make prediction using the model
    prediction = model.predict(img)
    
    # Assuming binary classification: 0 for DR, 1 for No DR
    if prediction[0][0] > 0.5:
        label = 'No DR'
    else:
        label = 'DR'
    
    # Draw the label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Diabetic Retinopathy Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
