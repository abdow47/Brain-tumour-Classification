import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import os

# Load the trained model (ensure BTP1.h5 is in the root of your Space)
MODEL_PATH = "BTP.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle error appropriately, maybe raise it or use a placeholder
    model = None

# Define class names (must match the order the model was trained on)
CLASS_NAMES = ["brain_glioma", "brain_menin", "brain_tumor"]
IMG_SIZE = 224

def predict_image(input_image: Image.Image):
    """
    Takes a PIL image, preprocesses it, and returns the prediction.
    Args:
        input_image: A PIL Image object.
    Returns:
        A dictionary mapping class names to confidence scores.
    """
    if model is None:
        return {"Error": "Model could not be loaded."}
    
    try:
        # Preprocess the image
        img = input_image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Note: We are not using preprocess_input or /255.0 based on the working script

        # Make prediction
        predictions = model.predict(img_array)[0] # Get the first (and only) prediction

        # Create a dictionary of class names and their probabilities
        confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        
        return confidences
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"Error": f"Prediction failed: {e}"}

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Brain MRI Scan"),
    outputs=gr.Label(num_top_classes=3, label="Prediction Results"),
    title="Brain Cancer Classification",
    description="Upload a brain MRI image to classify it as Glioma, Meningioma, or Tumor.",
    examples=[
        # Add paths to example images if you upload them to your Space
        # ["path/to/glioma_example.jpg"],
        # ["path/to/menin_example.jpg"],
        # ["path/to/tumor_example.jpg"]
    ],
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
