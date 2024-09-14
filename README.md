# Human Action Recognition using Deep Convolutional Neural Networks

## Overview

This project aims to build a deep learning system for **Human Action Recognition (HAR)** using images. The task is to develop a convolutional neural network (CNN) model that predicts the action performed by a person in an image and detects whether more than one person is present. The actions are categorized into 40 classes, making this a multi-class classification problem.

The solution involves training a custom network with additional layers on top of a pre-trained model to enhance prediction performance while addressing both action classification and person detection in a single network.

## Files Included

- **code.ipynb**: Jupyter notebook containing the training and evaluation code for the CNN model.
- **code.py**: Python script containing the implementation of the deep learning model.
- **mobilenet_model_adam_best.keras**: The best-performing trained model, saved in Keras format.
- **predictions.csv**: CSV file with predictions made on the provided test data.
- **best_model_hyperparameters.json**: JSON file containing the hyperparameters of the best-performing model.

## Project Goals

The primary goal of this project is to build a CNN model capable of:
1. Predicting the action class from 40 possible actions for a given image.
2. Detecting whether more than one person is present in the image, which helps determine prediction confidence.

## Model Architecture

- The model utilizes **MobileNet** as a base, followed by custom layers that are trained specifically for this task.
- The architecture was optimized using the **Adam optimizer** with a learning rate of **0.0001** and a dropout rate of **0.5** to reduce overfitting.
- The hyperparameters used to train the best model are stored in the `best_model_hyperparameters.json` file.

## How to Run

1. **Setup Environment**:
   - Ensure you have Python 3.x installed.
   - Install the required dependencies:
     ```bash
     pip install tensorflow keras pandas numpy
     ```

2. **Training the Model**:
   - Open and run the `code.ipynb` notebook to train the CNN model. The notebook covers loading the dataset, preprocessing, model design, training, and evaluation.

3. **Running Predictions**:
   - The trained model (`mobilenet_model_adam_best.keras`) can be used to make predictions on the test data:
     ```python
     from tensorflow import keras
     model = keras.models.load_model('mobilenet_model_adam_best.keras')
     # Load and preprocess your test data
     predictions = model.predict(test_data)
     ```

4. **Results**:
   - Predictions will be saved in `predictions.csv`. The format follows the expected structure for the task.

## Dataset

- The dataset consists of RGB images categorized into 40 classes, provided as part of the assignment.
- The training set includes labeled images, while a separate test set is provided for evaluating the model.

## Model Hyperparameters

The best model was tuned with the following hyperparameters:

- **Dropout Rate**: 0.5
- **L1 Regularization**: 0.0
- **L2 Regularization**: 1e-05
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Best Validation Loss**: 1.2868

These hyperparameters are saved in `best_model_hyperparameters.json`.

## Evaluation

- The model performance was evaluated using appropriate metrics such as accuracy and validation loss.
