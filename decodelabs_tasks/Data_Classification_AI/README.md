# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flower species using the K-Nearest Neighbors (KNN) algorithm.

## Overview

This project trains a KNN classifier on the classic Iris dataset and provides a Streamlit web interface for interactive species prediction based on flower measurements.

## Project Structure

```
Data_Classification_AI/
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
├── data/                # Dataset files
├── models/              # Trained model & scaler (pkl files)
├── notebooks/           # Jupyter notebooks for EDA
├── src/
│   ├── train.py         # Model training script
│   └── predict.py       # Prediction utilities
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python src/train.py
```

### Run the app
```bash
streamlit run app.py
```

## How It Works

1. **Input:** User enters sepal length, sepal width, petal length, and petal width
2. **Processing:** Features are scaled and passed to the KNN model
3. **Output:** Predicted species — Setosa, Versicolor, or Virginica
