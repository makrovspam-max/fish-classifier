import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Fish AI", layout="centered")

st.title("🐠 Fish Species Identifier")
st.write("Upload a photo of a fish to identify its species.")

# 1. Function to find and load the model
@st.cache_resource
def load_tflite_model():
    # This looks for the model file in the same folder as app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "fish_model.tflite")
    
    # Check if the file exists and display a helpful error if not
    if not os.path.exists(model_path):
        st.error(f"Model file not found! Expected it at: {model_path}")
        st.write("Files found in directory:", os.listdir(current_dir))
        return None

    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Initialize the model
interpreter = load_tflite_model()

# 2. Image Uploading
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("🔄 Classifying...")

    try:
        # 3. Preprocessing (Manual MobileNetV2 style)
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Scale pixels to [-1, 1] range (Common for TFLite models)
        img_array = (img_array / 127.5) - 1.0

        # 4. Run Inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        confidence = np.max(output_data)

        # 5. Display Results
        # Replace these labels with your actual fish species list in order!
        labels = ["Species A", "Species B", "Species C"] 
        
        if prediction < len(labels):
            st.success(f"Prediction: **{labels[prediction]}**")
            st.write(f"Confidence Score: {confidence:.2%}")
        else:
            st.info(f"Predicted Class ID: {prediction} (Update your labels list!)")

    except Exception as e:
        st.error(f"Error during classification: {e}")

elif interpreter is None:
    st.warning("The app cannot proceed without the model file.")
