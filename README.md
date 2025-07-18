# Multi‑Agent Anomaly Detection

**Détection d’anomalies réseau avec LSTM & explications LLM Mistral via FastAPI**

---

## 🎯 Contexte et Objectif

Ce projet combine :
1. Un **modèle LSTM** entraîné sur le dataset **UNSW‑NB15** pour détecter les fenêtres de flux réseau anormales.
2. Un **LLM local** (Mistral‑7B‑instruct Q4_K_M) pour générer des explications techniques et actionnables.
3. Une **API FastAPI** exposant deux endpoints :
   - `/predict` : renvoie la probabilité et la classe (normal/anomalie).  
   - `/analyze` : retourne la prédiction + des statistiques agrégées + une explication détaillée du LLM.

---


## ⚙️ Installation

1. **Cloner** le dépôt  
   ```bash
   git clone https://github.com/‹votre‑user›/multi-agent-anomaly-detection.git
   cd multi-agent-anomaly-detection

2. Créer et activer un environnement Python
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Installer les dépendances
pip install fastapi uvicorn llama-cpp-python numpy pandas scikit-learn tensorflow

4. Placer le modèle Mistral .gguf dans models/, et copier :
- lstm_model.weights.h5
- scaler.pkl
- feature_info.json


## 🚀 Lancement
uvicorn main:app --reload

→ L’API tourne sur http://127.0.0.1:8000

Docs interactives : http://127.0.0.1:8000/docs

