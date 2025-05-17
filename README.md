# Project Title: Palmprint-Based Biometric Authentication using Siamese Networks

## Project Overview

This project implements a biometric authentication system utilizing a Siamese neural network. The primary goal is to verify an individual's identity based on their unique palmprint characteristics. Siamese networks are particularly well-suited for this task as they learn to differentiate between input pairs, determining if they belong to the same class (i.e., the same individual) or different classes. This approach offers a secure and robust alternative to traditional authentication methods.

## Dataset

The model was developed using the **Tsinghua Palmprint Database**. This dataset contains 1,280 palmprint images collected from 80 distinct subjects. For each subject, images from two palms were captured, with eight different impressions taken per palm.
*(For this project, the entire dataset or a representative sample may have been used to train and evaluate the palmprint-based authentication model.)*

## Methodology

The development and implementation of the Siamese network for biometric authentication followed these key stages:

### 1. Data Loading and Preprocessing
* **Image Loading:** The raw palmprint images from the Tsinghua Palmprint Database were loaded.
* **Image Preprocessing:** A series of preprocessing steps were applied to the images. These steps are crucial for enhancing image quality, normalizing the data, and preparing it for the Siamese network. Common preprocessing techniques include resizing, grayscale conversion, contrast enhancement, and noise reduction. The specific steps undertaken were chosen based on the characteristics of the palmprint data and to optimize model performance.

### 2. Dataset Generation for Siamese Network
* To train a Siamese network, the dataset was transformed into pairs (or triplets) of images.
    * **Positive pairs:** Consisted of two different palmprint images from the same individual.
    * **Negative pairs:** Consisted of two palmprint images from different individuals.
* If a triplet network was used, triplets would consist of an anchor image, a positive image (same individual as anchor), and a negative image (different individual from anchor).
* A selection of these generated pairs/triplets was visualized to ensure correct formulation.

### 3. Data Splitting
* The generated dataset of image pairs/triplets was divided into **training and testing sets**. This separation allows for training the model on one subset of data and evaluating its generalization performance on unseen data.

### 4. Siamese Network Model Architecture and Training
* A **Siamese network** was constructed. This typically involves two identical sub-networks (twins) that process the input images from a pair.
    * The core of each sub-network usually consists of convolutional layers for feature extraction, followed by dense layers.
    * The outputs of these twin networks (feature vectors) are then combined, often using a distance metric (e.g., Euclidean distance, Manhattan distance) or a custom layer, to produce a similarity score.
* **Model Type:**
    * If a **twin network with binary cross-entropy** was used, the model was trained to predict whether a pair of images belongs to the same person (1) or different people (0).
    * If a **triplet network** was used, it was trained with a triplet loss function, which aims to minimize the distance between an anchor and a positive sample while maximizing the distance between the anchor and a negative sample by at least a certain margin.
* The model was trained on the prepared training set.

### 5. Model Evaluation
* The performance of the trained Siamese network was rigorously evaluated on the test set.
* Various metrics suitable for verification tasks were employed, such as:
    * Accuracy
    * Precision, Recall, F1-score (especially for distinguishing genuine vs. imposter pairs)
    * Area Under the ROC Curve (AUC-ROC)
    * Equal Error Rate (EER)

### 6. Performance Improvement (Iterative Process)
* If the initial test accuracy was not satisfactory, several strategies were explored to enhance model performance. This iterative process could include:
    * Trying different **network architectures** (e.g., varying the number/type of layers, activation functions).
    * Tuning **hyperparameters** (e.g., learning rate, batch size, optimizer).
    * Implementing techniques for **handling overfitting** (e.g., dropout, L2 regularization, data augmentation).

### 7. Biometric Authentication Application
* The final, optimized Siamese network model was utilized to perform biometric authentication.
* A system was developed where, given a user ID and a new palmprint image, the trained model verifies if the image indeed belongs to the claimed user.
* The results of several authentication attempts (both successful verifications and rejections of imposters) were displayed to demonstrate the practical application of the model.
