# DEEP-LEARNING-PROJECT

*COMPANY* : CODETECH IT SOLUTIONS

*NAME* : HRIDAYI HEMANT BHELLI

*INTERN ID* : CTIS8232

*DOMAIN* : DATA SCIENCE

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

#DESCRIPTION OF THE TASK

TITLE - Waste Classification Using Deep Learning for Image Recognition

This project focuses on developing an intelligent image classification system that can automatically identify and classify waste materials into three categories: Plastic, Paper, and Metal. The main objective of this project is to apply deep learning techniques to solve a real-world environmental problem by improving waste segregation and recycling efficiency. Proper waste classification is an important step in modern waste management systems because incorrect disposal of recyclable materials can lead to environmental pollution and resource loss.

In this project, a custom image dataset containing images of waste materials was used. The dataset was organized into separate folders for training and testing, with each category containing labeled images of plastic, paper, and metal waste. The images were preprocessed before training to improve the model’s learning ability. Preprocessing steps included image resizing, normalization, random rotation, and horizontal flipping. These techniques help improve model accuracy and reduce overfitting by exposing the model to different variations of the same object.

The deep learning model used in this project is based on ResNet18 (Residual Network), a powerful Convolutional Neural Network (CNN) architecture widely used for image classification tasks. Instead of training the model completely from scratch, transfer learning was used. Transfer learning allows us to use a pre-trained model that has already learned important image features from large datasets. The final classification layer of the model was modified to classify the three waste categories specific to this project. This approach significantly reduces training time and improves performance, especially when working with limited datasets.

The project was implemented using Python as the programming language. Several important libraries and tools were used during development. PyTorch was used for building and training the deep learning model. Torchvision was used for image transformations, loading datasets, and accessing pre-trained models. Matplotlib was used to create visualizations such as training loss graphs and confusion matrices. PIL (Python Imaging Library) was used for handling and testing individual images. The project was developed and executed using Visual Studio Code (VS Code), which served as the coding and debugging environment. The terminal inside VS Code and Windows PowerShell were used for running the Python scripts and installing required packages.

#OUTPUT

*CONFUSION MATRIX*
<img width="1912" height="1016" alt="Image" src="https://github.com/user-attachments/assets/8616ee13-395c-4b72-8d04-ce0228ad2110" />

*TEST ACCURACY*
<img width="1910" height="1018" alt="Image" src="https://github.com/user-attachments/assets/752c3e4a-3ea6-4e0c-88b0-75fd906984b0" />

*LOSS GRAPH*
<img width="1917" height="1022" alt="Image" src="https://github.com/user-attachments/assets/47fe8a45-0271-424e-9ce3-210260a858e5" />
