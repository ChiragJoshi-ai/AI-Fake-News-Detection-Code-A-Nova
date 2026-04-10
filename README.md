# 🧠✨ Fake News Detection API  
### *Truth Scanner for the Internet Age*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-⚡-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Enabled-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🌐 Overview

A sleek **Machine Learning-powered API** that analyzes news articles and predicts:

> 🟢 **REAL** or 🔴 **FAKE**

Built using:
- ⚡ FastAPI
- 🧠 TF-IDF + NLP features
- 🎯 Logistic Regression

---

## 🎬 How It Works

```
Input Article → Feature Extraction → ML Model → Prediction → Confidence + Signals
```

---

## 🧩 Features

- 🔍 Fake news detection
- 📊 Confidence scores
- 🧠 Linguistic signal extraction
- ⚡ Fast API responses
- 🌍 Frontend-ready (CORS enabled)

---

## 🧠 Model Insights

| Feature | Purpose |
|--------|--------|
| ❗ Exclamation Density | Detects sensational tone |
| 🔠 CAPS Ratio | Detects shouting / emphasis |
| 🧬 Lexical Diversity | Measures richness of text |
| 🚨 Red Flag Words | Flags suspicious phrases |

---

---

## 🏋️ Train the Model

```bash
python train_model.py
```

This generates:
```
detector.pkl
```

---

## ▶️ Run the API

```bash
uvicorn app:app --reload --port 8000
```

Visit:
👉 http://127.0.0.1:8000

---

## 📡 API Endpoints

### 🔮 POST `/predict`

#### Request
```json
{
  "title": "Breaking news title",
  "text": "Full article content..."
}
```

#### Response
```json
{
  "label": "REAL",
  "confidence": 92.3,
  "fake_prob": 7.7,
  "real_prob": 92.3,
  "signals": {
    "exclamation_density": 0.5,
    "caps_ratio": 2.1,
    "lexical_diversity": 78.4,
    "red_flag_phrases": []
  }
}
```


---

## 🧪 Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "title": "BREAKING: Scientists discover shocking truth",
  "text": "This is a shocking discovery they don’t want you to know!"
}'
```

---

## ⚠️ Disclaimer

- Model trained on **small synthetic dataset**
- Built for **learning & demonstration purposes**
- Not production-ready yet

---

⭐ If you like this project, give it a star!
