# Machine Learning Age Prediction Project

## Introduction

This GitHub repository contains the code and documentation for a machine learning project focused on predicting the age of individuals from facial images. The age prediction task spans a wide range, from 1 to 116 years, and involves handling unbalanced classes, for which a weighted loss function was employed during model training. Age groups were consolidated into ranges (e.g., 1-15, 16-25) to facilitate learning and evaluation.

## Project Structure

The project's code and resources are organized in a structured manner to allow for easy testing of individual modules or their combinations. The file structure is designed to promote modularity, code reusability, and straightforward experimentation.

```plaintext
project-root/
│
├── Data/
│   ├── Test/			# Test Data
│   ├── Train/			# Train Data
│   └── UTKFace/		# All Data
│
├── Models/
│   ├── model-??.??.?? ??;??	# Model #1 name
│   ├── model-??.??.?? ??;??	# Model #2 name
│   └── ...
│
├── Notebooks/
│   ├── Face Detection.ipynb   	# Face Detection notebook
│   ├── Training model.ipynb   	# Training model notebook
│   └── Train_test_split.ipynb 	# Train test split notebook
│
├── Structurized Files/
│   ├── AgeImageDataset.py 	# File contains dataset class
│   ├── Callbacks.py		# File contains all callbacks
│   ├── Constants.py	 	# File contains all constants
│   ├── main.py 		# Main file
│   ├── Model.py	 	# File contains the model
│   ├── Parameters.py	 	# File contains all parameters
│   ├── Preprocess.py	 	# File contains all preprocessing
│   ├── Saving.py	 	# File contains saving method
│   └── Training.py		# File contains training method
│
├── mlruns/
│   └─ experiment 1/
│   	├── run 1/		# Run #1 logs
│   	├── run 2/		# Run #2 logs
│   	└── ...        
│
├── requirements.txt		# Project dependencies
│
├── README.md			# Project README (You're here!)
│
└── .gitignore			# Gitignore file
```

## Model Training

For model training, I adopted a carefully chosen set of hyperparameters, which were determined through an iterative process of experimental learning. These parameters were fine-tuned by analyzing the results of model training using [MLflow](https://mlflow.org/), a versatile machine learning lifecycle management tool.

MLflow allowed us to track and visualize the training process, hyperparameter tuning, and performance metrics across various runs. Below is a screenshot illustrating some of the key insights gained from the MLflow experiments:

## Transfer Learning

In this project, I leveraged the power of transfer learning by comparing several popular pre-trained models, such as ResNet and DenseNet, alongside our custom deep learning model architecture. Transfer learning significantly accelerated the model's convergence and improved its performance on the age prediction task.

## Handling Class Imbalance

One of the unique challenges in this project was dealing with class imbalance, as age prediction naturally results in more samples in certain age ranges than others. To address this, a weighted loss function was utilized during training to give more emphasis to underrepresented age groups. This approach helped to balance the model's learning process and improve its performance.

## Docker container

I have containerized the entire project using Docker for easy deployment and reproducibility. The Docker image is available on Docker Hub and can be accessed and run in Linux environments. To build and run the Docker container, use the following commands:

```
# Build the Docker image
docker build -t age-estimation-image:1.0 .

# Run the Docker container
docker run -p 8501:8501 -it --device /dev/video0 age-estimation-image:1.0
```

Please note that this Docker setup is designed for Linux due to the simplicity of camera integration.
Probably you'll need to refresh the page once in a while, because of problems with webcam integration.

## Streamlit Web Application & Streamlit Community Cloud

To provide an intuitive interface for age prediction, I've developed a web application using [Streamlit](https://streamlit.io/). Streamlit is a Python library that simplifies the process of creating interactive web applications for machine learning models and data analysis. Our user-friendly web app allows users to upload images and receive age predictions in a user-friendly manner.

Our application is hosted on Streamlit Community Cloud, providing an accessible platform to interact with the age prediction model. You can access the hosted application at Streamlit Community Cloud.

## Conclusion

This project demonstrates an effective approach to age prediction from facial images, taking into account the wide range of ages and class imbalances. The structured codebase and the use of MLflow for experimentation make it easy to reproduce and extend the work. Feel free to explore the provided notebooks for data exploration and model evaluation, and refer to the `src` directory for detailed code implementations.

I welcome contributions and feedback to further enhance the accuracy and utility of this age prediction model. Thank you for visiting our project!