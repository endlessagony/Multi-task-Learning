# ABAW4 Competition MTL Feature Block Solution
This repository contains Jupyter Notebook(s) for the ABAW4 competition implementation of a custom solution with a "feature block" approach for MTL.

## Introduction
In this project, we propose a custom solution for the ABAW4 competition using an approach we call "feature block". 
This approach involves extracting features from each modality (e.g., facial expressions, audio, and text) and combining them into a single feature block. 
We then train a multi-task model on this feature block to predict the target emotions.

## Feature Block Architecture
Our "feature block" architecture consists of the following components:

1. Linear layer: This layer takes the input features and applies a linear transformation to them.

2. Batch Normalization: This layer normalizes the activations of the previous layer to improve the stability and performance of the model.

3. Leaky ReLU: This activation function introduces non-linearity to the model and prevents the "dying ReLU" problem.

4. Dropout: This regularization technique randomly drops out some of the activations to prevent overfitting and improve generalization.

5. Classifier Linear Layer: This layer applies a linear transformation to the output of the previous layer to produce the final predictions.

6. Activation Function: This function applies a non-linear transformation to the output of the classifier layer to produce the final predictions in the appropriate range.
