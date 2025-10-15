import torch
import torch.nn as nn
import numpy as np
import shap
import httpx
from fastapi import APIRouter, HTTPException
from fastapi import FastAPI, HTTPException, Query
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from datetime import datetime
# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hybrid CNN-LSTM Model
# -----------------------------
class CNNLSTM(nn.Module):
    def __init__(self, window, cnn_channels, kernel_size, lstm_hidden, lstm_layers, dropout=0.0, output_dim=12):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        c = self.conv1(x)
        c = self.relu(c)
        c = self.dropout(c)
        c = c.permute(0, 2, 1)
        out, _ = self.lstm(c)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)

# -----------------------------
# Config
# -----------------------------
CFG = {
    "window": 24,
    "cnn_channels": 64,
    "dropout": 0.5,
    "kernel_size":5,
    "lstm_hidden":128,
    "lstm_layers":1,
    "output_dim":12,
    "model_path":"app/best_w5ufnpb4.pt",
    # "data_api_url":"https://gee-spei-data-fetch.onrender.com/spei/woreda"
    "data_api_url":"http://45.134.226.201:8002/spei/woreda",
    "cache_file": "woreda_cache.json"
}

# Woreda mapping
AFAR_WOREDAS = sorted(["Elidar", "Bidu", "Kori"])   # alphabetically -> Bidu, Elidar, Kori
SOMALI_WOREDAS = sorted(["Godey", "Ewa", "Fik", "Hargele"])  # alphabetically -> Ewa, Fik,Godey, Hargele

# def map_woreda_name(woreda_name: str) -> str:
#     if woreda_name in AFAR_WOREDAS:
#         idx = AFAR_WOREDAS.index(woreda_name)
#         return f"Afar_{idx}"
#     elif woreda_name in SOMALI_WOREDAS:
#         idx = SOMALI_WOREDAS.index(woreda_name)
#         return f"Somali_{idx}"
#     else:
#         raise ValueError(f"Woreda '{woreda_name}' not recognized")


# -----------------------------
# Load trained model
# -----------------------------
model = CNNLSTM(
    window=CFG["window"],
    cnn_channels=CFG["cnn_channels"],
    kernel_size=CFG["kernel_size"],
    lstm_hidden=CFG["lstm_hidden"],
    lstm_layers=CFG["lstm_layers"],
    dropout=CFG["dropout"]
).to(device)

model.load_state_dict(torch.load(CFG["model_path"], map_location=device))
model.eval()

# -----------------------------
# Preprocessing functions
# -----------------------------
def create_sequences(data: List[float], window: int) -> List[np.ndarray]:
    sequences = []
    for i in range(len(data) - window + 1):
        sequences.append(np.array(data[i:i + window]))
    return sequences

def prepare_input(data: List[float], window: int) -> torch.Tensor:
    seqs = create_sequences(data, window)
    if not seqs:
        raise ValueError(f"Not enough data, need at least {window} points")
    last_seq = seqs[-1]
    return torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------------
# SHAP Explainer helper
# -----------------------------
def shap_explain(model, x, background, max_background=64):
    if len(background) > max_background:
        idx = np.linspace(0, len(background)-1, num=max_background, dtype=int).tolist()
        background = [background[i] for i in idx]

    background_X = torch.tensor(np.array(background), dtype=torch.float32).to(device)

    def model_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
            return model(x_t).detach().cpu().numpy()

    explainer = shap.KernelExplainer(model_fn, background_X.cpu().numpy())
    shap_values = explainer.shap_values(x.detach().cpu().numpy())

    if isinstance(shap_values, list):
        return [np.array(sv).tolist() for sv in shap_values]
    return np.array(shap_values).tolist()

def aggregate_predictions(points: list) -> list:
    """
    points: list of dicts, each with 'prediction' array
    returns: single array of length output_dim (mean across points)
    """
    if not points:
        return []
    
    # Stack predictions: shape = (num_points, output_dim)
    preds = np.array([pt["prediction"] for pt in points])
    
    # Compute mean along points axis -> shape = (output_dim,)
    aggregated = preds.mean(axis=0)
    
    return aggregated.tolist()

# -----------------------------
# Cache Persistence (with history)
# -----------------------------
CACHE: Dict[str, List[Dict[str, Any]]] = {}

def load_cache():
    global CACHE
    if os.path.exists(CFG["cache_file"]):
        with open(CFG["cache_file"], "r") as f:
            CACHE = json.load(f)
load_cache()
print("Loaded cache keys:", list(CACHE.keys()))

def save_cache():
    with open(CFG["cache_file"], "w") as f:
        json.dump(CACHE, f, indent=2)

# -----------------------------
# Background cache refresher
# -----------------------------
async def refresh_cache():
    all_woredas = AFAR_WOREDAS + SOMALI_WOREDAS
    timestamp = datetime.utcnow().isoformat()

    for woreda in all_woredas:
        try:
            mapped_name = woreda
            payload = {"woreda_name": mapped_name, "months": CFG["window"]}
            response = httpx.post(CFG["data_api_url"], json=payload, timeout=100)
            response.raise_for_status()
            points = response.json().get("points", [])

            results = []
            for pt in points:
                spei_series = pt.get("spei", [])
                if len(spei_series) < CFG["window"]:
                    continue
                try:
                    x = prepare_input(spei_series, CFG["window"])
                    with torch.no_grad():
                        prediction = model(x).detach().cpu().numpy().flatten().tolist()[:CFG["output_dim"]]
                    shap_values = shap_explain(model, x, create_sequences(spei_series, CFG["window"]))
                except Exception:
                    continue
                results.append({
                    "lat": pt.get("lat"),
                    "lon": pt.get("lon"),
                    "prediction": prediction,
                    "shap_values": shap_values
                })

            record = {
                "timestamp": timestamp,
                "aggregated_prediction": aggregate_predictions(results),
                "points": results
            }

            if woreda not in CACHE:
                CACHE[woreda] = []
            CACHE[woreda].append(record)

        except Exception as e:
            print(f"Error refreshing cache for {woreda}: {e}")

    save_cache()
# -----------------------------
# FastAPI setup
# -----------------------------
router = APIRouter()
@router.get("/analyze")
def analyze(
    woreda_name: str = Query(...),
    latest: bool = Query(default=True, description="If true (default), return only the latest record")
) -> Dict[str, Any]:
    if woreda_name not in CACHE or not CACHE[woreda_name]:
        raise HTTPException(status_code=404, detail=f"No cached results for {woreda_name}. Try again later.")
    return CACHE[woreda_name][-1] if latest else CACHE[woreda_name]
