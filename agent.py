# agent.py
import torch
import joblib
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from typing import List

class LSTMAnomalyAgent:
    def __init__(
        self,
        model_path: str = "lstm_model.weights.h5",
        scaler_path: str = "scaler.pkl",
        feature_info_path: str = "feature_info.json"
    ):
        # 1. Charger le scaler
        self.scaler = joblib.load(scaler_path)

        # 2. Charger les infos de colonnes
        with open(feature_info_path, "r") as f:
            info = json.load(f)
        self.columns = info["columns"]
        self.seq_len = info.get("sequence_length", 10)

        # 3. Colonnes catégorielles supposées
        self.cat_cols = ['proto', 'service', 'state']
        self.input_dim = len(self.columns)

        # 4. Recréer et charger le modèle
        self.model = self._build_model()
        self.model.load_weights(model_path)
        self.model.trainable = False

    def _build_model(self):
        model = Sequential([
            LSTM(20, return_sequences=True, input_shape=(self.seq_len, self.input_dim)),
            LSTM(20),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_sequence(self, sequence: List[dict]):
        df = pd.DataFrame(sequence)

        # Réordonner les colonnes selon le modèle
        df = df[self.columns]

        # Encoder les colonnes catégorielles
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes

        # Mise à l’échelle
        df_scaled = self.scaler.transform(df)

        # Reshape pour LSTM : (1, seq_len, features)
        return np.expand_dims(df_scaled, axis=0)

    def predict(self, sequence: List[dict]):
        input_seq = self.preprocess_sequence(sequence)
        prob = self.model.predict(input_seq)[0][0]
        return {"probability": float(prob), "class": int(prob > 0.5)}


# === TEST LOCAL ===
if __name__ == "__main__":
    # Exemple basé sur la ligne brute fournie
    example = {
        "dur": 0.121478,
        "proto": "tcp",
        "service": "-",
        "state": "FIN",
        "spkts": 6,
        "dpkts": 4,
        "sbytes": 258,
        "dbytes": 172,
        "rate": 74.08749,
        "sload": 252,
        "dload": 254,
        "sloss": 14158.94238,
        "dloss": 8495.365234,
        "sinpkt": 0,
        "dinpkt": 0,
        "sjit": 24.2956,
        "djit": 8.375,
        "swin": 30.177547,
        "stcpb": 11.830604,
        "dtcpb": 255,
        "dwin": 621772692,
        "tcprtt": 2202533631,
        "synack": 255,
        "ackdat": 0,
        "smean": 0,
        "dmean": 0,
        "trans_depth": 43,
        "response_body_len": 43,
        "ct_src_dport_ltm": 0,
        "ct_dst_sport_ltm": 0,
        "is_ftp_login": 1,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 1,
        "is_sm_ips_ports": 1
    }

    dummy_sequence = [example] * 10

    agent = LSTMAnomalyAgent()
    result = agent.predict(dummy_sequence)
    print("✅ Résultat de prédiction :", result)
