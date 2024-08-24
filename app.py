import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model('Model_age.h5')


def preprocess_img(img):
    # Resize image to the required input shape
    img = img.resize((75, 75), Image.LANCZOS)  # Use 75x75 for resizing
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array


st.title('Age Prediction from Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image and make prediction
    img_array = preprocess_img(img)
    prediction = model.predict(img_array)
    st.write(f"Predicted Age: {prediction[0][0]:.2f}")
