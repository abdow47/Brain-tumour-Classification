# Brain Cancer Classification Project

## Introduction

Hello everyone, this machine learning project focused on brain cancer classification. This project addresses a critical healthcare challenge by using deep learning to classify different types of brain cancer from medical images.

## Project Overview

The goal of this project is to develop a machine learning model that can accurately classify brain MRI scans into three categories: brain tumor, brain glioma, and brain meningioma. Early and accurate diagnosis of brain cancer is crucial for effective treatment planning, and machine learning can serve as a valuable tool to assist medical professionals in this process.

Data science transforms raw medical imaging data into valuable diagnostic information, helping healthcare providers make more informed decisions. Throughout this project, I've applied various techniques for data cleaning, exploratory analysis, feature engineering, model selection, and evaluation to create a robust classification system.

## Dataset Description

For this project, I've used the Multi-Cancer dataset from Kaggle, specifically focusing on the Brain Cancer subset. This dataset contains approximately 15,000 brain MRI images across three classes:
- Brain tumor (general)
- Brain glioma (a specific type of tumor that originates in the glial cells)
- Brain meningioma (tumors that form on membranes covering the brain and spinal cord)

The dataset required significant preprocessing as it contained raw medical images with varying quality, dimensions, and formats - making it an excellent candidate for demonstrating data cleaning techniques.


## Data Cleaning Phase

The data cleaning process was a critical step in this project. The raw MRI images required several preprocessing steps before they could be effectively used for model training:

1. First, I examined the dataset structure and identified that the images were organized in folders by cancer type, which helped in creating accurate labels.

2. I created a pandas DataFrame to organize the file paths and corresponding labels, making it easier to work with the dataset programmatically.

3. I checked for class imbalance and found that the dataset was perfectly balanced with 5,000 images per class, which is ideal for training without bias toward any particular class.

4. For image preprocessing, I implemented data augmentation techniques to enhance the training dataset and improve model generalization. This included random rotations, zooming, horizontal flips, and normalization of pixel values.

5. I also standardized image dimensions to ensure consistent input to the neural network, as medical images often come in varying sizes and resolutions.

All these cleaning steps were documented in the code with appropriate comments, making the process transparent and reproducible.

## Exploration Phase

For the exploratory data analysis, I investigated the dataset from multiple angles to gain insights that would inform my modeling approach:

1. I first examined the class distribution to confirm balance across the three cancer types, visualizing this with a bar chart.

2. I split the data into training (70%), validation (15%), and testing (15%) sets while maintaining the class distribution in each split.

3. I visualized sample images from each class to understand the visual characteristics that distinguish different cancer types.

4. I analyzed image properties such as dimensions, color channels, and pixel value distributions to inform preprocessing decisions.

5. I examined the relationship between image features and cancer types through various visualizations.

6. I created histograms of pixel intensities to understand the contrast and brightness patterns in different cancer types.

The exploration phase included both univariate analysis (examining single variables like class distribution) and bivariate analysis (relationships between variables). I used various visualization techniques including bar charts, histograms, scatter plots, heatmaps, and image grids to thoroughly explore the data.

## Visualization

Throughout the project, I created diverse and informative visualizations to represent the data and model performance:

1. Bar charts to show class distribution in the overall dataset and in each data split
2. Sample image grids to display representative examples from each cancer type
3. Confusion matrices to visualize model prediction accuracy across classes
4. Training and validation loss/accuracy curves to monitor model learning
5. ROC curves and precision-recall curves to evaluate model performance

Each visualization was carefully designed with appropriate labels, titles, and color schemes to make them easily interpretable. For example, when visualizing the confusion matrix, I used a normalized version with a color gradient to clearly show where misclassifications occurred most frequently.

## Feature Engineering and Selection

Feature engineering was a crucial aspect of improving model performance:

1. I implemented image augmentation as a form of feature engineering, creating transformed versions of the original images to help the model learn invariant features. This included:
   - Random rotations (up to 20 degrees)
   - Width and height shifts
   - Zoom variations
   - Horizontal flips
   - Brightness adjustments

2. I created a new feature by applying edge detection algorithms to highlight tumor boundaries, which provided additional discriminative information to the model.

3. For feature selection, I used an embedded method through the convolutional neural network architecture, which automatically learns the most relevant features from the images during training.

4. I implemented filter methods to select features based on their statistical properties, focusing on regions with high variance across cancer types.

These feature engineering and selection techniques significantly improved model performance by focusing on the most discriminative aspects of the brain MRI images.

## Algorithm Selection and Tuning

I experimented with multiple algorithms to find the best approach for this classification task:

1. Convolutional Neural Network (CNN): A custom architecture designed specifically for image classification
2. Transfer Learning with VGG16: Leveraging a pre-trained model on ImageNet with fine-tuning for our specific task
3. Transfer Learning with ResNet50: Another pre-trained model with residual connections for better gradient flow

For each algorithm, I conducted parameter tuning to optimize performance:

Parameter tuning is the process of finding the optimal hyperparameter values that maximize model performance. This is crucial because default parameters rarely yield the best results for specific datasets. For neural networks, parameters like learning rate, batch size, and network architecture significantly impact training dynamics and final performance.

I used GridSearchCV to systematically explore hyperparameter combinations for:
- Learning rate (ranging from 0.0001 to 0.01)
- Batch size (16, 32, 64)
- Dropout rates (0.2, 0.3, 0.5)
- Number of convolutional layers and filters
- Activation (ReLU)

After extensive experimentation, the transfer learning approach with ResNet50 and fine-tuning yielded the best performance, achieving over 95% accuracy on the validation set.

## Validation and Evaluation

To ensure robust model evaluation, I implemented a comprehensive validation strategy:

1. I used k-fold cross-validation to assess model performance across different data subsets, ensuring the results weren't dependent on a particular train-test split.

2. For model evaluation, I used multiple metrics:
   - Accuracy: Overall correct predictions
   - Precision: Ability to avoid false positives (crucial in medical diagnostics)
   - Recall: Ability to find all positive cases (vital for not missing cancer diagnoses)
   - F1-Score: Harmonic mean of precision and recall
   - ROC-AUC: Area under the Receiver Operating Characteristic curve

3. I discussed the importance of validation in ensuring model generalizability. Validation is critical because it helps detect overfitting and provides a realistic estimate of how the model will perform on unseen data. In medical applications like cancer detection, this is particularly important as the model needs to generalize well to new patients.

4. The final model achieved precision and recall values well above the required threshold of 0.3, with both metrics exceeding 0.9 for all three classes.

## Model Deployment

For deployment, I created a web application using Gradio that allows users to upload brain MRI images and receive instant classification results:

1. I saved the trained model and preprocessing transformers to files that could be loaded by the web application.

2. I developed a Gradio application (app.py) that:
   - Provides a user-friendly interface for image upload
   - Preprocesses uploaded images using the same pipeline as during training
   - Applies the trained model to make predictions
   - Displays results with confidence scores for each cancer type

3. The application was deployed to a web interface (as a huggingface space) and is accessible via the following link: [Brain Cancer Classifier App]([https://brain-cancer-classifier.herokuapp.com](https://huggingface.co/spaces/abdo47/tumour_classification))

4. The deployment process included thorough testing to ensure the application works correctly across different devices and browsers.

