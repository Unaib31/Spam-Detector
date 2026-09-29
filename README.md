# SMS Spam Detection Web Application

An interactive web application built with **Streamlit**, **TensorFlow/Keras**, and **scikit-learn** for real-time SMS message classification (Spam vs. Ham).

---

## 📌 Project Overview

This project provides an end-to-end Natural Language Processing (NLP) solution for identifying unwanted or fraudulent SMS text messages. By transforming raw text into numerical feature vectors using TF-IDF and processing them through a deep neural network, the app classifies incoming text messages into **SPAM** or **HAM** (legitimate) along with classification confidence scores and detailed token-level diagnostic analysis.

---

## ✨ Features

- **📩 SMS Detector Tab**: Interactive input box with example message buttons to classify text in real-time, complete with visual result cards and probability gauges.
- **🔬 NLP Analysis Tab**: Visual 8-step pipeline breakdown displaying lowercasing, cleaning, tokenization, stop-word removal, stemming, cleaned tokens, and message statistics (length, word count, digit count, special characters, uppercase ratio).
- **🧠 Model Information Tab**: Complete breakdown of the pre-trained neural network architecture, TF-IDF vectorisation setup, dataset distribution, and detailed explanations of common spam/ham text patterns.

---

## 🔄 NLP Preprocessing Pipeline

Every input message undergoes a systematic 5-stage NLP transformation prior to classification:

1. **Lowercasing**: Text is converted to lower-case characters.
2. **Regex Cleaning**: Non-alphabetic characters (numbers, punctuation, symbols) are removed (`[^a-z\s]`).
3. **Tokenization**: Cleaned text is split into individual word tokens.
4. **Stop-Word Removal**: Common English stop words (e.g., *the, is, at, which*) are filtered out using `nltk.corpus.stopwords`.
5. **Stemming**: Remaining tokens are reduced to their root forms using the NLTK `PorterStemmer`.

---

## 📊 TF-IDF Vectorisation

The preprocessed text tokens are converted into a high-dimensional numeric representation using a pre-fitted **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer**:

- **Vocabulary Size / Max Features**: 3,000 features
- **Output Format**: Dense floating-point array of shape `(1, 3000)` representing the relative importance of words across the corpus.

---

## 🧠 Neural Network Architecture

The classification backbone is a **Keras Sequential Deep Neural Network** trained on the TF-IDF feature vectors:

| Layer | Type | Output Shape | Parameters | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `Dense` | `(None, 128)` | 384,128 | ReLU |
| **1** | `Dropout` | `(None, 128)` | 0 | — (Rate: 0.5) |
| **2** | `Dense` | `(None, 64)` | 8,256 | ReLU |
| **3** | `Dense` | `(None, 1)` | 65 | Sigmoid |

- **Total Trainable Parameters**: `392,449`
- **Classification Threshold**: `0.50` (Output probability $\ge 0.50$ $\rightarrow$ **SPAM**, $< 0.50$ $\rightarrow$ **HAM**).

---

## 📁 Dataset Description

The classifier was trained on the benchmark **UCI SMS Spam Collection Dataset**:

- **Total Messages**: 5,574 labeled SMS messages
- **Legitimate (Ham)**: 4,827 messages (~86.6%)
- **Spam**: 747 messages (~13.4%)

---

## 🛠️ Project Structure

```
CaseStudy/
├── app.py                  # Streamlit web application interface & pipeline logic
├── spam_classifier.keras   # Pre-trained Keras Sequential neural network model
├── tfidf_vectorizer.pkl    # Pre-fitted scikit-learn TF-IDF vectorizer
├── requirements.txt        # Pinned Python package dependencies
├── README.md               # Project documentation
└── Assets/
    └── spam-app.png        # Web application preview screenshot
```

---

## 💡 Pre-trained Model Note

> **Note**: The pre-trained Keras model (`spam_classifier.keras`) and fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`) are already provided in the repository. **No model retraining is required** to run and evaluate the application.

---

## 📦 Requirements

- **Python**: 3.10+ (tested on Python 3.14)
- Key Python dependencies:
  - `streamlit`
  - `tensorflow`
  - `keras`
  - `scikit-learn`
  - `nltk`
  - `joblib`
  - `numpy`

Refer to [requirements.txt](file:///c:/My Projects/CaseStudy/requirements.txt) for exact package versions.

---

## 🚀 Local Setup & Run Instructions

1. **Clone or download** this repository to your local computer.
2. **Open a terminal** and navigate to the project root directory:
   ```bash
   cd CaseStudy
   ```
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the Streamlit application**:
   ```bash
   python -m streamlit run app.py
   ```
   *(Or using the direct CLI shortcut: `streamlit run app.py`)*
5. **Open your web browser** and navigate to `http://localhost:8501`.

---

## 📝 Example Usage

### Example 1: Spam Message
- **Input Text**: *"URGENT! You have won a £1000 guaranteed cash prize. Call 09066610571 now to claim your prize!"*
- **Predicted Class**: 🔴 **SPAM**
- **Confidence**: High spam probability ($\ge 0.50$)

### Example 2: Legitimate (Ham) Message
- **Input Text**: *"Hey, are we still meeting for lunch today at 1pm?"*
- **Predicted Class**: 🟢 **HAM**
- **Confidence**: Low spam probability ($< 0.50$)
