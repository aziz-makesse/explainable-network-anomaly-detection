# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from agent import LSTMAnomalyAgent
from llama_cpp import Llama
import numpy as np

# 1. Initialisation du LSTM Agent
agent = LSTMAnomalyAgent(
    model_path="lstm_model.weights.h5",
    scaler_path="scaler.pkl",
    feature_info_path="feature_info.json"
)

# 2. Chargement du LLM local Mistral
llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=0
)

app = FastAPI()

# Schéma d'entrée
class SequenceInput(BaseModel):
    sequence: List[Dict[str, Any]]

# Endpoint de prédiction brute (LSTM seule)
@app.post("/predict")
def predict_sequence(input_data: SequenceInput):
    """
    Retourne la prédiction binaire et la probabilité de la séquence fournie.
    """
    try:
        result = agent.predict(input_data.sequence)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint d'analyse (LSTM + Mistral)
@app.post("/analyze")
def analyze_sequence(input_data: SequenceInput):
    """
    Prédit avec LSTM puis génère une explication via Mistral,
    en analysant toute la séquence.
    """
    seq = input_data.sequence

    # 1) Prédiction LSTM
    pred = agent.predict(seq)

    # 2) Calcul de statistiques globales
    durs  = np.array([p["dur"] for p in seq])
    rates = np.array([p["rate"] for p in seq])
    spkts = np.array([p["spkts"] for p in seq])
    dpkts = np.array([p["dpkts"] for p in seq])
    http_count = sum(1 for p in seq if p.get("service") == "http")

    stats = {
        "count":      len(seq),
        "dur_mean":   float(durs.mean()),  "dur_min": float(durs.min()),  "dur_max": float(durs.max()),
        "rate_mean":  float(rates.mean()), "rate_min": float(rates.min()), "rate_max": float(rates.max()),
        "spkts_mean": float(spkts.mean()), "dpkts_mean": float(dpkts.mean()),
        "http_pct":   http_count / len(seq) * 100
    }

    # 3) Construction du prompt complet
    prompt = f"""
Tu es un expert en cybersécurité. Le modèle LSTM a analysé une séquence de {stats['count']} paquets et l’a classée comme 
**{'ANORMALE' if pred['class'] else 'NORMALE'}** (probabilité={pred['probability']:.3f}).

Statistiques de la séquence :
- dur (s) : moyenne={stats['dur_mean']:.3f}, min={stats['dur_min']:.3f}, max={stats['dur_max']:.3f}
- débit (rate) : moyenne={stats['rate_mean']:.3f}, min={stats['rate_min']:.3f}, max={stats['rate_max']:.3f}
- spkts moyenne={stats['spkts_mean']:.1f}, dpkts moyenne={stats['dpkts_mean']:.1f}
- pourcentage HTTP : {stats['http_pct']:.1f}%

Sur la base de ces éléments et du résultat du LSTM, explique :
1. Pourquoi le modèle l’a classée ainsi.
2. Quels patterns réseau (durées, débits, services) ressortent.
3. Les points clés qu’un analyste réseau devrait vérifier désormais.
"""

    # 4) Appel à Mistral
    try:
        output = llm(prompt, max_tokens=500, stop=["</s>"])
        explanation = output["choices"][0]["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return {
        "prediction": pred,
        "statistics": stats,
        "llm_explanation": explanation
    }
