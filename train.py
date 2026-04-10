"""
Fake News Detection - Model Training
Uses TF-IDF + NLP features + Gradient Boosting
"""

import pickle
import re
import string
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import FunctionTransformer
import warnings
warnings.filterwarnings('ignore')


def clean_text(text: str) -> str:
    """Lowercase, strip HTML artifacts, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # HTML tags
    text = re.sub(r'http\S+|www\S+', ' URL ', text)  # URLs → token
    text = re.sub(r'\d+', ' NUM ', text)           # digits → token
    text = re.sub(r'[^\w\s]', ' ', text)           # punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_heuristic_features(texts):
    """
    Hand-crafted signals that correlate with fake news:
      - exclamation / question mark density
      - ALL-CAPS ratio
      - average word length
      - type-token ratio (lexical diversity)
      - sentence length variance
    """
    feats = []
    for t in texts:
        words   = t.split()
        n_words = max(len(words), 1)
        chars   = list(t)

        excl_ratio  = t.count('!') / max(len(chars), 1)
        quest_ratio = t.count('?') / max(len(chars), 1)
        caps_ratio  = sum(1 for w in words if w.isupper() and len(w) > 1) / n_words
        avg_wlen    = np.mean([len(w) for w in words]) if words else 0
        sentences   = re.split(r'[.!?]', t)
        slens       = [len(s.split()) for s in sentences if s.strip()]
        slen_var    = np.var(slens) if slens else 0
        unique_r    = len(set(words)) / n_words   # type-token ratio

        feats.append([excl_ratio, quest_ratio, caps_ratio,
                      avg_wlen, slen_var, unique_r])
    return np.array(feats)


REAL_SNIPPETS = [
    "The Federal Reserve raised interest rates by 25 basis points today amid concerns about persistent inflation across major economies.",
    "Scientists have published new research showing a link between sleep deprivation and increased risk of cardiovascular disease.",
    "The city council voted 7-2 to approve the new transit budget, allocating funds for bus rapid transit expansion.",
    "A magnitude 5.4 earthquake struck 30 km north of the capital city early Tuesday morning with no major damage reported.",
    "The prime minister announced a new economic recovery package worth 50 billion dollars targeting small and medium enterprises.",
    "New satellite images reveal significant glacier retreat in the Arctic, consistent with climate models from the last decade.",
    "Researchers at the university completed a three-year study on antibiotic resistance in hospital settings across 12 countries.",
    "The trade deficit narrowed last month as exports of manufactured goods rose by 4.2 percent according to official data.",
    "Environmental agencies confirmed air quality levels returned to normal following last week's industrial incident.",
    "Health authorities issued updated guidelines for vaccination schedules based on the latest immunological research.",
    "The central bank kept interest rates unchanged citing stable inflation figures and moderate economic growth projections.",
    "Congress passed the infrastructure bill with bipartisan support following months of negotiation and amendment.",
    "Scientists detected unusual solar activity that may affect GPS and communication satellites over the next 48 hours.",
    "The annual report shows literacy rates have improved by 12 percent in rural districts following the education initiative.",
    "Court documents reveal the company paid 80 million dollars to settle the class action lawsuit without admitting wrongdoing.",
]

FAKE_SNIPPETS = [
    "BREAKING!! Government HIDING the TRUTH about the water supply — they are POISONING us all and the mainstream media won't say a word!!!",
    "Scientists ADMIT vaccines contain microchips to track every citizen — share this before it gets DELETED by the deep state!!!",
    "You WON'T BELIEVE what they found in tap water — doctors are BAFFLED and governments are PANICKING!!!",
    "EXCLUSIVE: Secret documents PROVE the moon landing was staged in a Hollywood studio — NASA insider CONFESSES!!!",
    "The CURE for cancer has been suppressed for 50 YEARS by Big Pharma — this simple herb DESTROYS tumors overnight!!!",
    "ALERT: 5G towers are CONFIRMED to spread the virus — telecom executives caught in MASSIVE cover-up!!!",
    "Globalists plan to reduce world population by 90% — leaked memo reveals the SHOCKING timeline!!!",
    "BANNED VIDEO: Politician caught on hot mic admitting to rigging elections — YouTube deleting this EVERYWHERE!!!",
    "Ancient pyramids were actually power plants — archaeologists FORBIDDEN from revealing the truth!!!",
    "URGENT SHARE: New law would make it ILLEGAL to grow your own food — they are coming for your gardens NEXT!!!",
    "TOP DOCTOR reveals hospitals are paid 13000 dollars for every COVID death they fake — BLOWING THE WHISTLE!!!",
    "PROOF: Birds aren't real — they are government surveillance drones charging on power lines!!!",
    "They are putting fluoride in water to make people STUPID and OBEDIENT — here's the TRUTH they hide!!!",
    "BOMBSHELL: Celebrity whistleblower exposes the satanic cult running Hollywood — names EXPOSED!!!",
    "Scientists SHOCKED as flat earth photos surface from NASA's own archive — the truth is OUT!!!",
]

def build_dataset():
    texts  = REAL_SNIPPETS + FAKE_SNIPPETS
    labels = [1] * len(REAL_SNIPPETS) + [0] * len(FAKE_SNIPPETS)   # 1=real, 0=fake
    return texts, labels


# ── Model Building ────────────────────────────────────────────────────────────

from scipy.sparse import hstack, csr_matrix

class FakeNewsDetector:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def fit(self, texts, labels):
        X_tfidf = self.tfidf.fit_transform(texts)
        X_heur  = csr_matrix(extract_heuristic_features(texts))
        X       = hstack([X_tfidf, X_heur])
        self.clf.fit(X, labels)

    def predict_proba(self, texts):
        X_tfidf = self.tfidf.transform(texts)
        X_heur  = csr_matrix(extract_heuristic_features(texts))
        X       = hstack([X_tfidf, X_heur])
        return self.clf.predict_proba(X)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)


def train_and_save():
    print("Loading dataset …")
    texts, labels = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    print("Training model …")
    model = FakeNewsDetector()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc:.2%}")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    out = "detector.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {out}")
    return model


if __name__ == "__main__":
    train_and_save()