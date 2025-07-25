import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time # For the progress spinner

# --- Configuration and Constants (must match training script) ---
IMAGE_SIZE = (224, 224) # MobileNetV2 input size
CLASS_NAMES = [
    'Bacterial_spot',
    'Early_blight',
    'healthy',
    'Late_blight',
    'Leaf_Mold',
    'powdery_mildew',
    'Septoria_leaf_spot',
    'Spider_mites Two-spotted_mite',
    'Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus'
]

# Ensure this path is correct relative to your app.py
# The model file is expected to be in a 'models' directory next to app.py
TFLITE_MODEL_PATH = 'models/tomato_disease_detection_model.tflite'
EXAMPLE_IMAGES_DIR = 'example_images' # Directory for example images

# --- Function to load and preprocess image ---
def preprocess_image(image_pil):
    """
    Preprocesses a PIL Image for model inference.
    Resizes, converts to numpy array, and applies MobileNetV2 preprocessing.
    """
    image_resized = image_pil.resize(IMAGE_SIZE)
    image_array = np.array(image_resized)
    # Expand dimensions to create a batch of 1 image (e.g., (1, 224, 224, 3))
    image_array = np.expand_dims(image_array, axis=0)
    # Apply MobileNetV2's specific preprocessing (scaling pixels to [-1, 1])
    # This is equivalent to tf.keras.applications.mobilenet_v2.preprocess_input
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

# --- Load the TFLite model ---
@st.cache_resource # Cache the model loading for efficiency
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model is in the correct directory.")
        st.stop() # Stop execution if model is not found
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        st.stop() # Stop execution if model loading fails

interpreter = load_tflite_model(TFLITE_MODEL_PATH)

# Get input and output tensor details (after successful loading)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Streamlit UI ---
st.set_page_config(
    page_title="Advanced Tomato Disease Detector",
    page_icon="🌿",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("Navigation & Information")
st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Go to",
    ("Disease Detection", "About This App", "How It Works")
)
st.sidebar.markdown("---")
st.sidebar.info("Developed by NO IDEA")



# --- Main Content Area ---

if selected_page == "Disease Detection":
    st.title("🌿 Tomato Leaf Disease Detection")
    st.markdown("Upload an image of a tomato leaf to instantly detect potential diseases. Our model is trained to identify 11 different conditions, including healthy leaves.")
    st.markdown("---")

    st.subheader("Try with an Example Image or Upload Your Own")
    
    example_image_files = []
    if os.path.exists(EXAMPLE_IMAGES_DIR):
        example_image_files = [f for f in os.listdir(EXAMPLE_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        example_image_files.sort() # Sort alphabetically for consistent order
    
    selected_example = None
    uploaded_file = None

    if example_image_files:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_example = st.selectbox("Select an example image:", ["Upload New Image"] + example_image_files)
        with col2:
            if selected_example != "Upload New Image":
                example_image_path = os.path.join(EXAMPLE_IMAGES_DIR, selected_example)
                example_image_pil = Image.open(example_image_path)
                st.image(example_image_pil, caption=f"Selected Example: {selected_example}", width=200)
            else:
                uploaded_file = st.file_uploader("Or, choose an image to upload:", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.file_uploader("Choose an image to upload:", type=["jpg", "jpeg", "png"])
        st.info(f"No example images found in the '{EXAMPLE_IMAGES_DIR}' directory. Consider adding some for a better user experience.")

    st.markdown("---")

    image_to_process = None
    image_caption = ""

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        image_caption = 'Uploaded Image'
    elif selected_example != "Upload New Image" and selected_example is not None:
        image_to_process = example_image_pil # Use the loaded PIL image from example
        image_caption = f"Selected Example: {selected_example}"


    if image_to_process:
        st.subheader("Image for Prediction:")
        st.image(image_to_process, caption=image_caption, use_column_width=True)
        st.write("")

        if st.button("Predict Disease"):
            with st.spinner('Classifying image...'):
                # Simulate a small delay for better UX if prediction is very fast
                # time.sleep(1)
                try:
                    # Preprocess the image
                    processed_image = preprocess_image(image_to_process)

                    # Perform inference
                    interpreter.set_tensor(input_details[0]['index'], processed_image)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])

                    # Get the predicted class and confidence
                    predicted_class_index = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])

                    predicted_class_name = CLASS_NAMES[predicted_class_index]

                    st.success(f"**Prediction:** {predicted_class_name}")
                    st.info(f"**Confidence:** {confidence:.2%}") # Format as percentage

                    st.write("---")
                    st.subheader("Detailed Probabilities:")
                    # Create a dictionary of class names and their probabilities
                    prob_dict = {CLASS_NAMES[i]: predictions[0][i] for i in range(len(CLASS_NAMES))}
                    # Sort probabilities in descending order
                    sorted_prob_items = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)

                    for class_name, prob in sorted_prob_items:
                        st.write(f"- **{class_name}:** {prob:.2%}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.warning("Please ensure the image is valid and try again.")
    else:
        st.info("Please upload an image or select an example to get started with disease detection.")

elif selected_page == "About This App":
    st.title("About the Tomato Disease Detector")
    st.markdown("""
    This application utilizes a deep learning model to identify various diseases affecting tomato leaves.
    It's built with **Streamlit** for the interactive user interface and **TensorFlow Lite** for efficient on-device inference.

    **Key Features:**
    * **Fast & Accurate:** Powered by a MobileNetV2 architecture, optimized for performance and accuracy on a wide range of devices.
    * **User-Friendly:** Intuitive interface with easy image upload or selection of examples for quick predictions.
    * **Comprehensive:** Capable of detecting 10 common tomato diseases and also identifying healthy leaves, covering a total of 11 classes.

    **Our Mission:**
    To empower farmers, home gardeners, and agricultural enthusiasts with a quick and accessible tool for preliminary diagnosis of tomato plant health issues. Early detection can lead to timely interventions, preventing crop loss and promoting healthier yields.
    """)
    st.subheader("Data Source:")
    st.write("The underlying machine learning model was meticulously trained on a diverse dataset of healthy and diseased tomato leaf images. This dataset is typically curated from publicly available resources like the PlantVillage project, which is widely used in agricultural AI research.")
    st.markdown("---")
    st.subheader("Future Enhancements:")
    st.write("""
    We are continuously working to improve this application by:
    * Expanding the number of detectable diseases.
    * Integrating recommendations for disease management.
    * Improving model accuracy and efficiency.
    """)


elif selected_page == "How It Works":
    st.title("How the Detector Works")
    st.markdown("""
    This application leverages advanced artificial intelligence techniques to analyze images of tomato leaves. Here's a simplified breakdown:

    1.  **Image Input:** Whether you upload your own image or select one of our examples, the first step is providing the leaf image to the system.
    2.  **Image Preprocessing:** Before the model can understand the image, it needs to be prepared. This involves:
        * **Resizing:** All images are uniformly resized to 224x224 pixels, the standard input size for our model.
        * **Normalization:** Pixel values are adjusted to a specific range (between -1 and 1), which helps the neural network process them effectively.
    3.  **Transfer Learning with MobileNetV2:**
        * Our model is based on **MobileNetV2**, a powerful yet lightweight convolutional neural network (CNN) architecture developed by Google.
        * It was initially trained on a massive dataset of general images (**"ImageNet"**), allowing it to learn to recognize a wide variety of features (edges, textures, shapes).
        * Through **transfer learning**, we adapted this pre-trained model by adding a few custom layers and retraining it specifically on images of tomato leaves. This allows it to recognize subtle patterns indicative of different diseases.
    4.  **Prediction by TensorFlow Lite:**
        * The model performs its analysis and outputs a set of probabilities, indicating how likely the image is to belong to each of the 11 defined classes (e.g., Bacterial Spot, Healthy, Early Blight).
        * **TensorFlow Lite** is used for this step. It's a specialized version of TensorFlow optimized for efficient execution on mobile and edge devices, ensuring quick predictions even on less powerful hardware.
    5.  **Result Display:** The application then presents the most probable disease and a confidence score. It also shows the probabilities for other potential diseases, giving you a comprehensive understanding.
    """)
    st.subheader("Technology Stack:")
    st.markdown("""
    * **Backend Machine Learning:** TensorFlow/Keras
    * **Model Optimization:** TensorFlow Lite (`.tflite`)
    * **Web Application Framework:** Streamlit (Python)
    * **Image Processing:** Pillow (PIL)
    """)
    st.markdown("---")
   
    