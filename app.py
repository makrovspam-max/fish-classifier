import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Fish AI Classifier", page_icon="🐠")

st.title("🐠 Fish Species Identifier")
st.write("Upload a photo of a fish to identify its species.")

# --- SMART MODEL LOADER ---
@st.cache_resource
def load_my_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_files = os.listdir(current_dir)
    
    # 1. Try the specific name first
    target_name = "fish_brain.tflite"
    model_path = os.path.join(current_dir, target_name)
    
    # 2. If missing, look for ANY file ending in .tflite
    if not os.path.exists(model_path):
        tflite_files = [f for f in all_files if f.lower().endswith('.tflite')]
        if tflite_files:
            model_path = os.path.join(current_dir, tflite_files[0])
            st.info(f"Auto-detected model file: **{tflite_files[0]}**")
        else:
            st.error("❌ No .tflite model file found in your GitHub repository!")
            st.write("Files currently visible to the app:", all_files)
            return None

    try:
        # Check size to ensure it's not a tiny 'pointer' file
        if os.path.getsize(model_path) < 10000:
            st.error("❌ The model file on GitHub is too small. Please re-upload the 8.5MB file.")
            return None
            
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading the model: {e}")
        return None

# Initialize the interpreter
interpreter = load_my_model()

# --- USER INTERFACE ---
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    # Open and show the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Photo', use_column_width=True)
    
    st.write("⚙️ Analyzing...")

    try:
        # Preprocessing: Resize to 224x224 (Standard for most TFLite models)
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize: Scale pixels to [-1, 1]
        img_array = (img_array / 127.5) - 1.0

        # Run the AI
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_index = np.argmax(output_data)
        confidence = np.max(output_data)

        # --- UPDATE YOUR LABELS HERE ---
        # Put your species names in the correct order!
        labels = ["Species A", "Species B", "Species C"] 
        
        if prediction_index < len(labels):
            st.success(f"Prediction: **{labels[prediction_index]}**")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.warning(f"Predicted Class ID: {prediction_index}")
            st.info("Tip: Update the 'labels' list in your code to show species names.")

    except Exception as e:
        st.error(f"Classification error: {e}")

elif interpreter is None:
    st.warning("⚠️ App is waiting for a valid 8.5MB .tflite file to be uploaded to GitHub.")
        st.info(f"Class ID: {prediction} (Add more names to your 'labels' list!)")

elif interpreter is None:
    st.warning("Waiting for a valid model file to be uploaded to GitHub...")
