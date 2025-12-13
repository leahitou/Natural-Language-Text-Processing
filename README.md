# Hate Speech Detection using Deep Learning

This project implements a hate speech detection system using deep learning techniques. The model classifies tweets into three categories: **hate speech**, **offensive language**, or **neither**.

## 🧠 Model Architecture
- **Embedding Layer**: Converts text to dense vectors.
- **Stacked LSTM Layers**: Capture sequential dependencies in text.
- **Dense Output Layer**: 3-class classification using softmax.

## 📊 Dataset
- **Source**: `labeled_data.csv` https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset?resource=download
- **Size**: 24,783 tweets
- **Classes**:
  - `0` → Hate Speech
  - `1` → Offensive Language
  - `2` → Neither

## 🛠️ Preprocessing Pipeline
1. **Cleaning**: Remove non-alphabetic characters, extra spaces.
2. **Lemmatization**: Reduce words to base forms using spaCy.
3. **Stopword Removal**: Eliminate common words.
4. **Text Encoding**: One-hot encoding + padding to fixed length.
5. **Handling Imbalance**: SMOTE oversampling applied to minority classes.

## 📈 Performance
- **Accuracy**: ~88.34%
- **Precision/Recall/F1-Score**: Detailed in classification report.
- **Confusion Matrix**: Visualized using seaborn.

## 🧪 Training Details
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Train/Test Split**: 80/20

## 🚀 How to Run
1. Upload `labeled_data.csv` to Colab or local environment.
2. Open and run `Hate_Speech_Detection.ipynb`.
3. Ensure required libraries are installed:
   ```bash
   pip install pandas spacy tensorflow scikit-learn imbalanced-learn seaborn matplotlib
   python -m spacy download en_core_web_sm

📌 Dependencies
- Python 3.x
- TensorFlow / Keras
- spaCy
- scikit-learn
- imbalanced-learn
- pandas, numpy, matplotlib, seaborn
