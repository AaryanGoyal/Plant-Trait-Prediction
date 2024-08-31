# Kaggle Competition: Plant-Trait-Prediction

## Overview
This project is a submission for the CS-480 Spring 2024 Kaggle competition. The task involves predicting multiple target variables based on a combination of image embeddings extracted using a pre-trained ViT (Vision Transformer) model and additional ancillary data.

## Project Structure
The repository contains the following files:
- **main_notebook.ipynb**: The complete Jupyter Notebook with all code and steps required for data processing, model training, and prediction generation.

## Approach
### 1. Data Loading
- Ancillary data is loaded from CSV files.
- Image paths are generated based on the dataset.

### 2. Image Embedding Extraction
- A pre-trained DINOv2 ViT model is used to extract image embeddings for all images in the training and test datasets.
- The images are preprocessed using a series of augmentations to ensure robustness.

### 3. Feature Engineering
- Image embeddings are combined with the ancillary features to create a comprehensive feature set.
- These features are then split into training and validation sets.

### 4. Model Training
- Three different models were trained: XGBoost, LightGBM, and CatBoost. Each model was trained independently for the six target variables.
- A meta-model (Linear Regression) was trained to combine the predictions from the three models to improve the overall accuracy.

### 5. Evaluation
- The meta-model’s performance was evaluated using the R2 score on the validation set.
- Different combinations of the models were tested to identify the best ensemble strategy.

### 6. Prediction and Submission
- The meta-model was used to generate final predictions for the test set.
- A submission file was generated in the required format.

## Dependencies
To run the notebook, you'll need the following Python libraries:
- torch
- torchvision
- timm
- PIL (Pillow)
- numpy
- pandas
- tqdm
- catboost
- lightgbm
- xgboost
- scikit-learn

You can install the required libraries using:
```bash
pip install torch torchvision timm Pillow numpy pandas tqdm catboost lightgbm xgboost scikit-learn
