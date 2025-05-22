import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# Force CPU usage to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set page configuration
st.set_page_config(
    page_title="Brain Cancer Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Brain Cancer Classification")
st.markdown("""
This application uses a deep learning model to classify brain MRI scans into three categories:
- Brain Tumor
- Brain Glioma
- Brain Meningioma

Upload a brain MRI scan to get a prediction.
""")

# Sidebar for model information
with st.sidebar:
    st.header("About the Model")
    st.info("""
    This model was trained on the Multi-Cancer dataset from Kaggle, 
    specifically focusing on brain MRI scans. It uses a deep learning 
    architecture based on EfficientNet to classify 
    images into three categories of brain cancer.
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload a brain MRI scan image
    2. Wait for the model to process the image
    3. View the prediction results and confidence scores
    """)

# Function to load the model
@st.cache_resource
def load_classification_model():
    """
    Load the trained model.
    Returns:
        The loaded model
    """
    try:
        # Update the path to point to the Downloads folder and use the correct filename
        model_path = os.path.expanduser("~/Downloads/BTP1.h5")
        
        if not os.path.exists(model_path):
            st.warning(f"Model file not found at {model_path}. Please check the file location.")
            st.info("In a real deployment, you would need to include your trained model file.")
            return None
        
        # Load model directly without compile=False parameter
        model = load_model(model_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(img):
    """
    Preprocess the image to match the model's expected input.
    Args:
        img: PIL Image object
    Returns:
        Preprocessed image as numpy array
    """
    # Resize to match model input size
    img = img.resize((224, 224))
    
    # Convert to array using the same method as the working code
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # No need for manual normalization, use preprocess_input instead
    # This handles the preprocessing specific to EfficientNet
    # img_array = preprocess_input(img_array)
    
    return img_array

# Function to make predictions
def predict(model, img_array):
    """
    Make prediction using the model.
    Args:
        model: Loaded model
        img_array: Preprocessed image array
    Returns:
        Prediction results
    """
    try:
        # Class labels in the same order as the working code
        class_labels = ['brain_glioma', 'brain_menin', 'brain_tumor']
        
        # Get predictions
        predictions = model.predict(img_array)
        
        # For a single image, get the index of the class with highest probability
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]
        
        # Get confidence scores for all classes
        confidence_scores = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
        
        return {
            'predicted_class': predicted_class,
            'confidence_scores': confidence_scores,
            'confidence_percentage': float(np.max(predictions[0])) * 100
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Function to display prediction results
def display_prediction_results(results, img):
    """
    Display the prediction results in a user-friendly format.
    Args:
        results: Dictionary containing prediction results
        img: Original image
    """
    if results is None:
        return
    
    # Display the predicted class with confidence percentage
    st.subheader("Prediction Result")
    st.markdown(f"**Predicted Class:** {results['predicted_class'].replace('_', ' ').title()} with {results['confidence_percentage']:.1f}% confidence")
    
    # Display confidence scores
    st.subheader("Confidence Scores")
    
    # Create a bar chart for confidence scores
    confidence_data = pd.DataFrame({
        'Class': [k.replace('_', ' ').title() for k in results['confidence_scores'].keys()],
        'Confidence': list(results['confidence_scores'].values())
    })
    
    # Sort by confidence score
    confidence_data = confidence_data.sort_values('Confidence', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        confidence_data['Class'], 
        confidence_data['Confidence'],
        color=['#2196F3', '#4CAF50', '#FF9800']
    )
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height*100:.1f}%',
            ha='center', 
            va='bottom'
        )
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Confidence Score')
    ax.set_title('Prediction Confidence by Class')
    
    # Display the plot
    st.pyplot(fig)
    
    # Add interpretation
    st.subheader("Interpretation")
    max_confidence = max(results['confidence_scores'].values())
    
    if max_confidence > 0.7:
        st.success(f"The model is highly confident in its prediction of {results['predicted_class'].replace('_', ' ').title()}.")
    elif max_confidence > 0.4:
        st.warning(f"The model has moderate confidence in its prediction. Consider seeking a second opinion.")
    else:
        st.error("The model has low confidence in its prediction. This image may be unclear or contain features the model wasn't trained on.")

# Image upload and display section
st.header("Upload Brain MRI Scan")

# File uploader widget
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

# Display and process the uploaded image
if uploaded_file is not None:
    # Create columns for image display and information
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display the uploaded image
        st.subheader("Uploaded Image")
        
        # Read the image
        image_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_bytes))
        
        # Display the image
        st.image(img, caption='Uploaded MRI Scan', use_column_width=True)
        
        # Display image information
        st.info(f"""
        **Image Information:**
        - Format: {img.format}
        - Size: {img.size}
        - Mode: {img.mode}
        """)
    
    with col2:
        st.subheader("Image Processing")
        
        # Show a spinner while processing
        with st.spinner("Processing image..."):
            # Preprocess the image
            preprocessed_img = preprocess_image(img)
            
            # Display the preprocessed image
            st.markdown("**Preprocessed Image (224x224)**")
            
            # Convert back to PIL for display
            display_img = Image.fromarray((preprocessed_img[0]).astype(np.uint8))
            st.image(display_img, caption='Preprocessed for Model Input', use_column_width=True)
            
            st.success("Image preprocessed successfully!")
    
    # Add a divider
    st.markdown("---")
    
    # Prediction section
    st.header("Prediction")
    
    # Button to trigger prediction
    if st.button("Run Prediction"):
        # Load the model
        model = load_classification_model()
        
        # If model is available, make prediction
        if model is not None:
            with st.spinner("Running prediction..."):
                # Make prediction
                results = predict(model, preprocessed_img)
                
                # Display results
                display_prediction_results(results, img)
        else:
            # If model is not available, use simulation for demonstration
            st.warning("Could not load the model. Please check the model file path and format.")
else:
    # Display sample images or instructions when no file is uploaded
    st.info("Please upload a brain MRI scan image to get started.")
    
    # Display sample images
    st.subheader("Sample Images")
    st.markdown("""
    Here are examples of the types of images the model was trained on:
    
    *Note: These are just examples. In a real deployment, you would include actual sample images.*
    """)
    
    # Create columns for sample images
    col1, col2, col3 = st.columns(3)
    
    # Placeholder for sample images
    with col1:
        st.markdown("**Brain Tumor Example**")
        st.markdown("*Sample image would appear here*")
    
    with col2:
        st.markdown("**Brain Glioma Example**")
        st.markdown("*Sample image would appear here*")
    
    with col3:
        st.markdown("**Brain Meningioma Example**")
        st.markdown("*Sample image would appear here*")
