# Hate Speech Detection – NLP Project

This project implements a hate speech detection system using Natural Language
Processing and Machine Learning techniques. The goal is to classify tweets into
three categories: Hate Speech, Offensive Language, or Neither.

---

## Dataset

The dataset consists of labeled tweets with three classes:
- **0**: Hate Speech  
- **1**: Offensive Language  
- **2**: Neither  

---

## 🧠 Model Architecture
- **Embedding Layer**: Converts text to dense vectors.
- **Stacked LSTM Layers**: Capture sequential dependencies in text.
- **Dense Output Layer**: 3-class classification using softmax.

--

## Methodology

1. Exploratory Data Analysis (EDA) was performed to understand class distribution.
2. Text data was transformed using **TF-IDF vectorization**.
3. Multiple machine learning models were trained and benchmarked.
4. Logistic Regression was selected for deployment.
5. A Flask web application was built for real-time predictions.
6. Unit tests were implemented to verify model behavior.

---

## Model Performance

The final model achieved strong accuracy and F1-score using TF-IDF features and
Logistic Regression. Benchmarking was performed against SVM and Naive Bayes
models.

---

📌 Dependencies
- Python 3.x
- TensorFlow / Keras
- spaCy
- scikit-learn

--

## How to Run the Application

```bash
pip install -r requirements.txt
python app.py

- imbalanced-learn
- pandas, numpy, matplotlib, seaborn
