import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
from PIL import Image

# Load and compile the model
model_path = 'forexmodel.h5'
loaded_model = load_model(model_path, compile=False)
loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
st.write("Model loaded and compiled successfully")

# Function to preprocess the image
def preprocess_image(image):
    img_size = (400, 400)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, 0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to predict the class of the image
def predict(image, model):
    img_array = preprocess_image(image)
    st.write(f"Preprocessed image array shape: {img_array.shape}")
    prediction = model.predict(img_array)
    st.write(f"Raw prediction: {prediction}")
    return prediction

# Streamlit interface
st.title("Forex Trading Signal Predictor")

# Upload images
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Classifying..."):
        prediction = predict(image, loaded_model)
        
        # Assuming train_gen.class_indices 0 is BUY and 1 is SELL
        classes = ['BUY', 'SELL']
        buy_prob = prediction[0][0] * 100
        sell_prob = prediction[0][1] * 100
        predicted_class = classes[np.argmax(prediction)]
        
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: BUY {buy_prob:.2f}% | SELL {sell_prob:.2f}%")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info("""
    This app uses a trained deep learning model to predict Forex trading signals.
    Upload an image to get a prediction of BUY or SELL.
""")
