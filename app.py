"""
Fake News Detection — Inference API
Run: uvicorn app:app --reload --port 8000
"""

import pickle
import re
import string
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "detector.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── API setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Fake News Detector", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ───────────────────────────────────────────────────────────────────
class ArticleRequest(BaseModel):
    text: str
    title: Optional[str] = ""


class PredictionResponse(BaseModel):
    label: str          # "REAL" | "FAKE"
    confidence: float   # 0-100
    fake_prob: float
    real_prob: float
    signals: dict


# ── helpers ───────────────────────────────────────────────────────────────────
def extract_signals(text: str) -> dict:
    words   = text.split()
    n       = max(len(words), 1)
    excl    = round(text.count('!') / max(len(text), 1) * 100, 2)
    caps    = round(sum(1 for w in words if w.isupper() and len(w) > 1) / n * 100, 2)
    unique  = round(len(set(words)) / n * 100, 2)

    red_flags = re.findall(
        r'\b(SHOCKING|BOMBSHELL|EXCLUSIVE|BREAKING|COVER.UP|DEEP STATE|'
        r'THEY DON.T WANT|BANNED|EXPOSED|WAKE UP|SHARE BEFORE|DELETED)\b',
        text.upper()
    )
    return {
        "exclamation_density": excl,
        "caps_ratio": caps,
        "lexical_diversity": unique,
        "red_flag_phrases": list(set(red_flags)),
    }


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(req: ArticleRequest):
    combined = f"{req.title} {req.text}".strip()
    if len(combined) < 20:
        raise HTTPException(400, "Article text is too short (minimum 20 characters).")

    probs     = model.predict_proba([combined])[0]
    fake_p, real_p = float(probs[0]), float(probs[1])
    label     = "REAL" if real_p >= 0.5 else "FAKE"
    confidence= round(max(fake_p, real_p) * 100, 1)
    signals   = extract_signals(combined)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        fake_prob=round(fake_p * 100, 1),
        real_prob=round(real_p * 100, 1),
        signals=signals,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}