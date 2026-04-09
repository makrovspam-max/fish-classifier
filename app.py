import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Fish Classifier", page_icon="🐠")

st.title("🐠 Fish Species Identifier")
st.write("Upload a photo to see what species of fish it is.")

# --- MODEL LOADING ---
@st.cache_resource
def load_my_model():
    # We are using the new 'fish_brain.tflite' name here
    model_path = os.path.join(os.path.dirname(__file__), "fish_brain.tflite")
    
    if not os.path.exists(model_path):
        st.error(f"Cannot find 'fish_brain.tflite' in the GitHub folder.")
        return None

    try:
        # Diagnostic: Check if file is actually a model or just a 1KB text file
        size = os.path.getsize(model_path)
        if size < 10000: # Less than 10KB is likely a 'pointer' error
            st.error(f"The model file is too small ({size} bytes). It might be a Git LFS pointer.")
            return None
            
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

interpreter = load_my_model()

# --- USER INTERFACE ---
uploaded_file = st.file_uploader("Upload a fish photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Target Fish', use_column_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]

    # Run AI
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    
    # --- EDIT THESE LABELS ---
    # Put your fish names here in the exact order they were trained!
    labels = ["Species 1", "Species 2", "Species 3"] 
    
    if prediction < len(labels):
        st.success(f"Result: {labels[prediction]}")
    else:
        st.info(f"Class ID: {prediction} (Add more names to your 'labels' list!)")

elif interpreter is None:
    st.warning("Waiting for a valid model file to be uploaded to GitHub...")
