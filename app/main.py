from fastapi import FastAPI
from .api.index import router as index

app = FastAPI(title="SPEI + Model API")

app.include_router(index)

@app.get("/")
def root():
    return {"status": "ok", "message": "SPEI + CNN-LSTM API running"}
