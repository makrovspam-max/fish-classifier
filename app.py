import os
import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

@st.cache_resource
def load_tflite_model():
    # This finds the directory where app.py is sitting
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "fish_model.tflite")
    
    # Check if the file actually exists before trying to load it
    if not os.path.exists(model_path):
        st.error(f"File not found at: {model_path}")
        st.write("Files in folder:", os.listdir(current_dir))
        return None

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Define the Fish types (MUST BE ALPHABETICAL)
class_names = ['Angelfish', 'Guppy', 'Platy']

# 3. The UI
uploaded_file = st.file_uploader("Take a photo of your fish...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Analyzing this fish...', use_container_width=True)
    
    # Preprocess the image for the model
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    # This line is crucial for MobileNetV2 models
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # 4. Run the TFLite "Brain"
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # 5. Show Results
    score = np.max(predictions)
    result = class_names[np.argmax(predictions)]
    
    st.success(f"I'm {score*100:.1f}% sure that's a **{result}**!")
