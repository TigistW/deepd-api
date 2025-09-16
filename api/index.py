import torch
import torch.nn as nn
import numpy as np
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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
        # x: [batch_size, window]
        x = x.unsqueeze(1)  # [batch_size, 1, window]
        c = self.conv1(x)
        c = self.relu(c)
        c = self.dropout(c)
        c = c.permute(0, 2, 1)  # [batch_size, window, cnn_channels]
        out, _ = self.lstm(c)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)

# -----------------------------
# Config
# -----------------------------

CFG = {
    "batch_size": 64,
    "cnn_channels": 64,
    "dropout": 0.5,
    "epochs": 60,
    "kernel_size":5,
    "lstm_hidden" :128,
    "lstm_layers":1,
    "lr": 0.00066188,
    "patience":5,
    "target_col": "mean",
    "train_frac":0.6,
    "val_frac": 0.2,
    "window":24,

}

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

model.load_state_dict(torch.load("best_w5ufnpb4.pt", map_location=device))
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

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()

class AnalyzeRequest(BaseModel):
    data: List[float]
    background: Optional[List[float]] = None
    window: Optional[int] = None
    steps: Optional[int] = None

@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    w = req.window or CFG["window"]
    steps = req.steps or CFG["output_dim"]

    try:
        x = prepare_input(req.data, w)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prediction
    with torch.no_grad():
        prediction = model(x).detach().cpu().numpy().flatten().tolist()
    prediction = prediction[:steps]

    # Background for SHAP
    if req.background and len(req.background) >= w:
        background = create_sequences(req.background, w)
    else:
        background = create_sequences(req.data, w)

    if not background:
        raise HTTPException(status_code=400, detail="Not enough background data for SHAP")

    shap_values = shap_explain(model, x, background)

    return {
        "window": w,
        "steps": steps,
        "input_window": x.squeeze(0).detach().cpu().numpy().tolist(),
        "prediction": prediction,
        "shap_values": shap_values
    }
