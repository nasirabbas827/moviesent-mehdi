# moviesent‑mehdi  

A sentiment analysis project that predicts movie review polarity using both classic machine‑learning (Logistic Regression) and deep‑learning (LSTM) approaches. The repository contains the full data pipeline, trained models, a Flask web interface, and a Jupyter notebook for exploration and reproducibility.  

---  

## Overview  

- **Goal:** Classify movie reviews as *positive* or *negative* (or neutral) using textual features.  
- **Approach:**  
  1. TF‑IDF vectorisation + Logistic Regression.  
  2. Tokenisation + LSTM network (Keras/TensorFlow).  
- **Deliverables:** Trained models, a reusable Flask API, and an interactive notebook that walks through data cleaning, feature engineering, training, and evaluation.  

---  

## Features  

| ✅ | Description |
|---|-------------|
| **Data preprocessing** | CSV loading, missing‑value handling, text cleaning, and train‑test split. |
| **Feature extraction** | TF‑IDF vectoriser (`tfidf_vectorizer.pkl`) for classic ML; Keras `Tokenizer` (`tokenizer.pkl`) for LSTM. |
| **Two modelling pipelines** | Logistic Regression (`logistic_regression_model.pkl`) and LSTM (`lstm_model.h5`). |
| **Web UI** | Simple Flask app (`app.py`) with an HTML front‑end (`templates/index.html`) for real‑time predictions. |
| **Reproducibility** | All code in a Jupyter notebook (`moviesent_mehdi.ipynb`) and a `requirements.txt` file for environment setup. |
| **Exportable artifacts** | Pre‑trained models and vectorisers are shipped for immediate inference without retraining. |

---  

## Tech Stack  

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.x |
| **Data handling** | pandas, numpy |
| **ML / DL** | scikit‑learn, TensorFlow/Keras |
| **Web framework** | Flask |
| **Front‑end** | HTML (Bootstrap optional) |
| **Environment** | Jupyter Notebook, virtualenv/conda |
| **Version control** | Git (GitHub) |

---  

## Installation  

1. **Clone the repository**  

   ```bash
   git clone https://github.com/<your‑username>/moviesent-mehdi.git
   cd moviesent-mehdi
   ```

2. **Create a virtual environment** (recommended)  

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Download the dataset** – the CSV is already included under `Dataset/Dataset.csv`.  

5. **Verify the installation**  

   ```bash
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

---  

## Usage  

### 1. Run the Jupyter Notebook  

```bash
jupyter notebook Notebook\ File/moviesent_mehdi.ipynb
```

The notebook walks through:

- Data loading (`Dataset/Dataset.csv`)  
- Text cleaning & tokenisation  
- TF‑IDF vectorisation and Logistic Regression training  
- LSTM model building, training, and evaluation  
- Saving the artefacts (`*.pkl`, `*.h5`)  

Feel free to modify cells and re‑run experiments.

### 2. Launch the Flask web app  

```bash
export FLASK_APP=app.py