import torch
import torch.nn as nn
import numpy as np
import shap
import httpx
from fastapi import APIRouter, HTTPException
from fastapi import FastAPI, HTTPException, Query
from typing import Dict, Any, List, Optional

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
    "data_api_url":"https://gee-spei-data-fetch.onrender.com/spei/woreda"
}

# Woreda mapping
AFAR_WOREDAS = sorted(["Elidar", "Bidu", "Kori"])   # alphabetically -> Bidu, Elidar, Kori
SOMALI_WOREDAS = sorted(["Godey", "Fik", "Hargele"])  # alphabetically -> Fik, Godey, Hargele

def map_woreda_name(woreda_name: str) -> str:
    if woreda_name in AFAR_WOREDAS:
        idx = AFAR_WOREDAS.index(woreda_name)
        return f"Afar_{idx}"
    elif woreda_name in SOMALI_WOREDAS:
        idx = SOMALI_WOREDAS.index(woreda_name)
        return f"Somali_{idx}"
    else:
        raise ValueError(f"Woreda '{woreda_name}' not recognized")


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
# FastAPI setup
# -----------------------------
router = APIRouter()
@router.get("/analyze")
def analyze(woreda_name: str = Query(..., description="Name of the woreda")) -> Dict[str, Any]:
    
    try:
        mapped_name = map_woreda_name(woreda_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Fetch data from SPEI API
    try:
        payload = {"woreda_name": mapped_name, "months": CFG["window"]}
        response = httpx.post(CFG["data_api_url"], json=payload, timeout=100)
        response.raise_for_status()
        data_json = response.json()
        points = data_json.get("points", [])
        if not points:
            raise HTTPException(status_code=500, detail="No points returned from SPEI API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from API: {e}")

    results = []
    for pt in points:
        spei_series = pt.get("spei", [])
        if len(spei_series) < CFG["window"]:
            continue
        try:
            x = prepare_input(spei_series, CFG["window"])
            with torch.no_grad():
                prediction = model(x).detach().cpu().numpy().flatten().tolist()[:CFG["output_dim"]]
            background = create_sequences(spei_series, CFG["window"])
            shap_values = shap_explain(model, x, background)
        except Exception as e:
            continue
        results.append({
            "lat": pt.get("lat"),
            "lon": pt.get("lon"),
            "prediction": prediction,
            "shap_values": shap_values
        })

    # return {
    #     "woreda_name": woreda_name,
    #     "points": results
    # }
    # Aggregate predictions for the woreda
    aggregated_prediction = aggregate_predictions(results)

    return {
        "woreda_name": woreda_name,
        "aggregated_prediction": aggregated_prediction,
        "points": results  # optional: keep individual points if needed
    }

# import torch
# import torch.nn as nn
# import numpy as np
# import shap

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional

# # -----------------------------
# # Device setup
# # -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -----------------------------
# # Hybrid CNN-LSTM Model
# # -----------------------------
# class CNNLSTM(nn.Module):
#     def __init__(self, window, cnn_channels, kernel_size, lstm_hidden, lstm_layers, dropout=0.0, output_dim=12):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
#                             num_layers=lstm_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(lstm_hidden, output_dim)

#     def forward(self, x):
#         # x: [batch_size, window]
#         x = x.unsqueeze(1)  # [batch_size, 1, window]
#         c = self.conv1(x)
#         c = self.relu(c)
#         c = self.dropout(c)
#         c = c.permute(0, 2, 1)  # [batch_size, window, cnn_channels]
#         out, _ = self.lstm(c)
#         last = out[:, -1, :]
#         last = self.dropout(last)
#         return self.fc(last)

# # -----------------------------
# # Config
# # -----------------------------

# CFG = {
#     "batch_size": 64,
#     "cnn_channels": 64,
#     "dropout": 0.5,
#     "epochs": 60,
#     "kernel_size":5,
#     "lstm_hidden" :128,
#     "lstm_layers":1,
#     "lr": 0.00066188,
#     "patience":5,
#     "target_col": "mean",
#     "train_frac":0.6,
#     "val_frac": 0.2,
#     "window":24,

# }

# # -----------------------------
# # Load trained model
# # -----------------------------
# model = CNNLSTM(
#     window=CFG["window"],
#     cnn_channels=CFG["cnn_channels"],
#     kernel_size=CFG["kernel_size"],
#     lstm_hidden=CFG["lstm_hidden"],
#     lstm_layers=CFG["lstm_layers"],
#     dropout=CFG["dropout"]
# ).to(device)

# model.load_state_dict(torch.load("best_w5ufnpb4.pt", map_location=device))
# model.eval()

# # -----------------------------
# # Preprocessing functions
# # -----------------------------
# def create_sequences(data: List[float], window: int) -> List[np.ndarray]:
#     sequences = []
#     for i in range(len(data) - window + 1):
#         sequences.append(np.array(data[i:i + window]))
#     return sequences

# def prepare_input(data: List[float], window: int) -> torch.Tensor:
#     seqs = create_sequences(data, window)
#     if not seqs:
#         raise ValueError(f"Not enough data, need at least {window} points")
#     last_seq = seqs[-1]
#     return torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

# # -----------------------------
# # SHAP Explainer helper
# # -----------------------------
# def shap_explain(model, x, background, max_background=64):
#     if len(background) > max_background:
#         idx = np.linspace(0, len(background)-1, num=max_background, dtype=int).tolist()
#         background = [background[i] for i in idx]

#     background_X = torch.tensor(np.array(background), dtype=torch.float32).to(device)

#     def model_fn(x_np: np.ndarray) -> np.ndarray:
#         with torch.no_grad():
#             x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
#             return model(x_t).detach().cpu().numpy()

#     explainer = shap.KernelExplainer(model_fn, background_X.cpu().numpy())
#     shap_values = explainer.shap_values(x.detach().cpu().numpy())

#     if isinstance(shap_values, list):
#         return [np.array(sv).tolist() for sv in shap_values]
#     return np.array(shap_values).tolist()

# # -----------------------------
# # FastAPI setup
# # -----------------------------
# app = FastAPI()

# class AnalyzeRequest(BaseModel):
#     data: List[float]
#     background: Optional[List[float]] = None
#     window: Optional[int] = None
#     steps: Optional[int] = None

# @app.post("/analyze")
# def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
#     w = req.window or CFG["window"]
#     steps = req.steps or CFG["output_dim"]

#     try:
#         x = prepare_input(req.data, w)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     # Prediction
#     with torch.no_grad():
#         prediction = model(x).detach().cpu().numpy().flatten().tolist()
#     prediction = prediction[:steps]

#     # Background for SHAP
#     if req.background and len(req.background) >= w:
#         background = create_sequences(req.background, w)
#     else:
#         background = create_sequences(req.data, w)

#     if not background:
#         raise HTTPException(status_code=400, detail="Not enough background data for SHAP")

#     shap_values = shap_explain(model, x, background)

#     return {
#         "window": w,
#         "steps": steps,
#         "input_window": x.squeeze(0).detach().cpu().numpy().tolist(),
#         "prediction": prediction,
#         "shap_values": shap_values
#     }

