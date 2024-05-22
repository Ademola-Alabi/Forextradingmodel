import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('updateforexmodel.h5')

# Define image preprocessing function
def preprocess_image(image):
    img_size = (400, 400)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, *img_size, 3))
    return img_array

# Define prediction function
def predict(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    classes = ["SELL", "BUY"]
    return classes[class_idx]

# Streamlit interface
st.title("Forex Prediction Model")

# Upload images
uploaded_files = st.file_uploader("Choose image files", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Classifying..."):
            label = predict(image, model)
            st.write(f"Prediction: {label}")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info("""
    This app uses a trained deep learning model to predict Forex signals.
    Upload an image to get a prediction of BUY or SELL.
""")
