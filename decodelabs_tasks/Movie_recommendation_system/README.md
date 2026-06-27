# 📚 Book Recommendation System

A content-based recommendation system that suggests similar books using cosine similarity.

## Overview

This project preprocesses book data from Goodreads, computes feature similarities, and provides personalized book recommendations through a Streamlit web interface.

## Project Structure

```
Movie_recommendation_system/
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
├── data/                # Goodreads book dataset
├── models/              # Saved model artifacts
├── notebooks/           # Jupyter notebooks for analysis
├── src/
│   ├── preprocess.py    # Data loading & preprocessing
│   └── recommender.py   # Recommendation engine
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## How It Works

1. **Select** a book from the dropdown list
2. **Click** "Recommend" to get suggestions
3. **View** recommended books with authors and similarity scores
