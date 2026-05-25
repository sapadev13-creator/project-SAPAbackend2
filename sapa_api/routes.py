import base64
import hashlib
import logging
import os

import pandas as pd
import tweepy
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from oauthlib.common import generate_token
from pydantic import BaseModel
from requests_oauthlib import OAuth2Session

from sapa_api import state
from sapa_api.config import (
    AUTH_URL,
    DEVICE,
    FRONTEND_URL,
    TOKEN_URL,
    TWITTER_CLIENT_ID,
    TWITTER_REDIRECT_URI,
    TWITTER_SCOPES,
)
from sapa_api.excel_io import (
    build_excel_rows,
    dataframe_to_excel_bytes,
    excel_buffer_to_base64,
)
from sapa_api.pipeline import compute_ocean_prediction, run_ocean_pipeline
from sapa_api.twitter import fetch_user_tweets

router = APIRouter()
logging.basicConfig(level=logging.INFO)


class TextInput(BaseModel):
    text: str


@router.get("/")
def root():
    return {
        "service": "SAPA OCEAN API",
        "device": DEVICE,
        "subtraits": state.LEXICAL_SIZE,
        "status": "OK",
    }


@router.get("/auth/twitter/login")
def twitter_login(request: Request):
    code_verifier = generate_token(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode("utf-8")

    oauth = OAuth2Session(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_REDIRECT_URI,
        scope=TWITTER_SCOPES,
    )
    state_token = generate_token(32)
    authorization_url, _ = oauth.authorization_url(
        AUTH_URL,
        state=state_token,
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )
    request.session["oauth_state"] = state_token
    request.session["code_verifier"] = code_verifier
    return RedirectResponse(authorization_url)


@router.get("/auth/twitter/callback")
def twitter_callback(request: Request, code: str, state: str):
    if state != request.session.get("oauth_state"):
        raise HTTPException(400, "Invalid OAuth state")

    oauth = OAuth2Session(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_REDIRECT_URI,
        scope=TWITTER_SCOPES,
        state=state,
    )
    token = oauth.fetch_token(
        TOKEN_URL,
        code=code,
        code_verifier=request.session["code_verifier"],
        client_secret=os.getenv("TWITTER_CLIENT_SECRET"),
    )
    request.session["twitter_access_token"] = token["access_token"]
    return RedirectResponse(url=f"{FRONTEND_URL}?twitter=success", status_code=302)


@router.get("/auth/twitter/me")
def twitter_me(request: Request):
    access_token = request.session.get("twitter_access_token")
    if not access_token:
        raise HTTPException(401, "Not authenticated")
    me = tweepy.Client(access_token).get_me()
    return {"username": me.data.username}


@router.get("/predict/twitter/check")
def twitter_check(request: Request):
    return {"logged_in": bool(request.session.get("twitter_access_token"))}


@router.post("/predict/twitter")
def predict_from_twitter(request: Request):
    access_token = request.session.get("twitter_access_token")
    if not access_token:
        raise HTTPException(401, "Twitter not authenticated")

    twitter_text = fetch_user_tweets(access_token)
    if not twitter_text.strip():
        raise HTTPException(404, "No tweets found")

    return run_ocean_pipeline(
        text=twitter_text,
        username=tweepy.Client(access_token).get_me().data.username,
    )


@router.post("/predict/twitter/profile")
def predict_other_profile(data: dict, request: Request):
    try:
        profile_url = data.get("profile_url")
        if not profile_url:
            raise HTTPException(400, "Missing profile_url")

        username = profile_url.rstrip("/").split("/")[-1].replace("@", "")
        logging.info(f"Fetching tweets for {username}")

        bearer = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer:
            raise HTTPException(500, "TWITTER_BEARER_TOKEN not set in .env")

        app_client = tweepy.Client(bearer_token=bearer)
        user = app_client.get_user(username=username)
        logging.info(f"User found: {user.data}")
        tweets = app_client.get_users_tweets(
            id=user.data.id,
            max_results=10,
            exclude=["retweets", "replies"],
        )
        if not tweets.data:
            raise HTTPException(404, "No tweets found")

        text = " ".join(t.text for t in tweets.data)
        return run_ocean_pipeline(text=text, username=username)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in /predict/twitter/profile: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}") from e


def _process_excel_upload_sync(
    file_bytes: bytes,
    filename: str,
    with_profile: bool,
    *,
    fast: bool,
    batch_size: int,
    include_row_details: bool,
    include_charts: bool,
):
    from io import BytesIO

    from sapa_api.batch_processing import process_texts_batch

    if not filename.endswith(".xlsx"):
        raise HTTPException(400, "File harus berformat .xlsx")

    df = pd.read_excel(BytesIO(file_bytes))
    if "text" not in df.columns:
        raise HTTPException(400, "Excel harus memiliki kolom 'text'")

    items: list[tuple[int, str]] = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        if text.strip():
            items.append((int(idx), text))

    results, timing = process_texts_batch(
        items,
        fast=fast,
        batch_size=max(1, min(batch_size, 64)),
        include_charts=include_charts,
        include_heavy_details=include_row_details,
    )

    df_detail = build_excel_rows(results)
    profile_summary = None
    if with_profile:
        from sapa_api.viz import aggregate_ocean_profile

        profile_summary = aggregate_ocean_profile(results)

    excel_buffer = dataframe_to_excel_bytes(df_detail, profile_summary=profile_summary)
    excel_b64 = excel_buffer_to_base64(excel_buffer)

    payload = {
        "status": "success",
        "processing": timing,
        "excel": {
            "filename": (
                "ocean_profile_result.xlsx" if with_profile else "ocean_result.xlsx"
            ),
            "content_base64": excel_b64,
        },
    }

    if with_profile:
        payload["total_text"] = len(results)
        payload["profile_summary"] = profile_summary
        if include_row_details:
            payload["row_results"] = results
        return payload

    payload["total_rows"] = len(results)
    if include_row_details:
        payload["results"] = results
    return payload


async def _process_excel_upload(
    file: UploadFile,
    with_profile: bool,
    *,
    fast: bool = True,
    batch_size: int = 16,
    include_row_details: bool = False,
    include_charts: bool = False,
):
    import asyncio

    body = await file.read()
    return await asyncio.to_thread(
        _process_excel_upload_sync,
        body,
        file.filename or "upload.xlsx",
        with_profile,
        fast=fast,
        batch_size=batch_size,
        include_row_details=include_row_details,
        include_charts=include_charts,
    )


@router.post("/predict/excel")
async def predict_from_excel(
    file: UploadFile = File(...),
    fast: bool = True,
    batch_size: int = 16,
    include_row_details: bool = False,
    include_charts: bool = False,
):
    return await _process_excel_upload(
        file,
        with_profile=False,
        fast=fast,
        batch_size=batch_size,
        include_row_details=include_row_details,
        include_charts=include_charts,
    )


@router.post("/predict/excel/profile")
async def predict_from_excel_profile(
    file: UploadFile = File(...),
    fast: bool = True,
    batch_size: int = 16,
    include_row_details: bool = False,
    include_charts: bool = False,
):
    return await _process_excel_upload(
        file,
        with_profile=True,
        fast=fast,
        batch_size=batch_size,
        include_row_details=include_row_details,
        include_charts=include_charts,
    )


@router.post("/predict")
def predict(data: TextInput):
    pred = compute_ocean_prediction(data.text)
    return {
        "input_text": pred.get("input_text", data.text),
        "text_normalized": pred.get("text_normalized", data.text),
        "typo_corrections": pred.get("typo_corrections", []),
        "highlighted_text": pred["highlighted_text"],
        "prediction_raw": pred["raw"],
        "prediction_adjusted": pred["adjusted"],
        "dominant_trait": pred["dominant"],
        "text_intent": pred.get("text_intent", "neutral"),
        "dominant_keyword_category": pred.get("dominant_keyword_category"),
        "personality_profile": pred["personality_profile"],
        "ontology_analysis": {
            "coverage_percent": pred["coverage"],
            "active_subtraits": pred["subtraits"],
            "semantic_subtraits": pred["semantic_subtraits"],
        },
        "lexical_evidence": pred["evidence"],
        "semantic_ontology_matches": pred["semantic_matches"],
        "ontology_expansion_candidates": pred["ontology_expansion_candidates"],
        "crisis_level": pred.get("crisis_level", "none"),
        "analysis_mode": pred.get("analysis_mode", "trait"),
        "confidence_scale": pred.get("confidence_scale", 1.0),
        "meaningful_token_count": pred.get("meaningful_token_count", 0),
        "disclaimer": pred.get("disclaimer"),
        "big_five_constructs": pred.get("evidence", {}).get("big_five_constructs", []),
        "sentiment_modifiers": pred.get("sentiment_modifiers", {}),
        "explanation": pred["explanation"],
        "suggestion": pred["suggestion"],
    }
