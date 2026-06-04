import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.logger_setup import logger
from sapa_api.config import (
    SESSION_SECRET_KEY,
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_CLIENT_ID,
)
from sapa_api.routes import router
from sapa_api.startup import load_ontology_and_model

torch.set_num_threads(1)

if not TWITTER_API_KEY or not TWITTER_API_SECRET:
    raise RuntimeError("TWITTER_API_KEY or TWITTER_API_SECRET not set in .env")
if not TWITTER_CLIENT_ID:
    raise RuntimeError("TWITTER_CLIENT_ID is not set in .env")

logger.info("FastAPI app starting...")

app = FastAPI(
    title="SAPA OCEAN API",
    description="Ontology-aware Indonesian Personality Prediction",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sapadev.id",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="lax",
    https_only=False,
)
app.include_router(router)


@app.on_event("startup")
def startup_event():
    logger.info("Startup loading ontology & model")
    load_ontology_and_model()
