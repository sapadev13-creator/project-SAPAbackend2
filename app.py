import os
import torch
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from torch import nn

MODEL_DIR = "./model"
OUTPUT_DIR = "./outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

app = FastAPI(
    title="OCEAN Personality Regression API",
    description="Predict Big Five Personality Traits (Likert 0–5)",
    version="1.1.0",
)


class TinyBERTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_DIR)
        self.regressor = nn.Sequential(
            nn.Linear(312, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.regressor(cls)


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TinyBERTRegressor().to(DEVICE)
model.regressor.load_state_dict(
    torch.load(f"{MODEL_DIR}/regressor.pt", map_location=DEVICE)
)
model.eval()
print("✅ Model loaded on", DEVICE)


class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: List[str]


def _scores_dict(pred) -> dict:
    return dict(zip(LABELS, (round(float(x), 3) for x in pred)))


def predict_texts(texts: List[str]) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    return np.clip(outputs.cpu().numpy(), 0.0, 5.0)


def _read_upload_dataframe(file: UploadFile) -> pd.DataFrame:
    if file.filename.endswith(".csv"):
        return pd.read_csv(file.file)
    return pd.read_excel(file.file)


@app.get("/")
def root():
    return {
        "status": "OK",
        "model": "TinyBERT OCEAN Regression",
        "scale": "Likert 0–5",
    }


@app.post("/predict")
def predict_single(data: TextInput):
    return _scores_dict(predict_texts([data.text])[0])


@app.post("/predict-batch")
def predict_batch(data: BatchInput):
    preds = predict_texts(data.texts)
    results = [_scores_dict(p) for p in preds]
    return {"count": len(results), "results": results}


@app.post("/predict-file")
def predict_file(file: UploadFile = File(...), text_column: str = "text"):
    if not file.filename.endswith((".xlsx", ".csv")):
        raise HTTPException(status_code=400, detail="File harus CSV atau XLSX")

    df = _read_upload_dataframe(file)
    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom '{text_column}' tidak ditemukan",
        )

    preds = predict_texts(df[text_column].astype(str).tolist())
    for i, label in enumerate(LABELS):
        df[label] = preds[:, i].round(3)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"prediction_{file.filename.replace('.', '_')}.xlsx",
    )
    df.to_excel(output_path, index=False)

    return {
        "status": "success",
        "rows": len(df),
        "output_file": output_path,
    }
