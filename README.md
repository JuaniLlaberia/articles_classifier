# Articles Classifier v0.0.1 (Beta)

A project to classify news articles by it's category (e.g. politics, economy, etc.) using machine learning and natural language processing techniques.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Architecture & Approach](#architecture--approach)  
- [Tech Stack](#tech-stack)  
- [Getting Started](#getting-started)  

---

## Overview

**Articles Classifier** is a machine learning / NLP project for automatic classification of articles.  
Given an input text (e.g. news articles), the system predicts a category based on a trained model.  

It can be used for:  
- Content tagging  
- Topic categorization  
- Recommendation systems  
- Any pipeline needing structured labels from articles

---

## Features

- Train a classification model on labeled article data ([data used](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news))
- Predict categories for new articles  
- Preprocessing pipeline: cleaning, embeddings  
- Evaluation with metrics and confusion matrix  
- Modular architecture – easy to extend with new datasets or models

Comming in next version:
- Unit, Integration and E2E testing
- Improvements in the model (We are still researching and trying to get better results in all categories)

---

## Architecture & Approach

1. **Data ingestion & preprocessing**  
   - Load labeled dataset  
   - Clean articles 
   - Extract features (embeddings)  

2. **Model training & selection**  
   - Train ML models (XGBoost, but we also got good results using Logistic Regression)  
   - Hyperparameter tuning  
   - Cross-validation
   - Save model for future predictions

3. **Inference & prediction**  
   - Predict labels for unseen text  
   - Return category + probabilities of other categories 

---

## 📦 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/JuaniLlaberia/articles_classifier
   cd articles_classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Agent locally (FastAPI):

   ```bash
   python .
   ```

With that you can start classifying articles using this trained model, if you want to further explore or train, you will need to download the [data](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news) as is not provided in the repo.
