from collections import defaultdict

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from app.logger_setup import logger
from sapa_api import state
from sapa_api.config import DEVICE, DEVICE_INFO, HF_REPO, ONTOLOGY_CSV, ONTOLOGY_EMB
from sapa_api.model_def import OceanModel
from sapa_api.fuzzy_match import build_fuzzy_index
from sapa_api.semantic import prepare_embedding_index


def load_ontology_and_model():
    state.ontology_df = pd.read_csv(ONTOLOGY_CSV)
    if state.ontology_df is None or state.ontology_df.empty:
        raise RuntimeError("Ontology CSV kosong / gagal dibaca")

    state.ontology_df["tokens"] = state.ontology_df["lexeme"].astype(str).apply(
        lambda x: x.split("_")
    )
    if "strength" not in state.ontology_df.columns:
        state.ontology_df["strength"] = 1.0

    state.SUBTRAITS = sorted(state.ontology_df["sub_trait"].dropna().unique())
    state.LEXICAL_SIZE = len(state.SUBTRAITS)
    if state.LEXICAL_SIZE == 0:
        raise RuntimeError("LEXICAL_SIZE = 0, ontology bermasalah")

    state.subtrait2id = {s: i for i, s in enumerate(state.SUBTRAITS)}
    state.LEXICON = defaultdict(list)
    state.LEXICON_PATTERNS = []
    state.TOKEN_TO_PATTERN_IDS = defaultdict(list)
    for _, row in state.ontology_df.iterrows():
        sub_trait = row["sub_trait"]
        tokens = [t for t in row["tokens"] if t]
        tok_set = set(tokens)
        pat = {
            "sub_trait": sub_trait,
            "tokens": tok_set,
            "strength": float(row["strength"]),
            "lexeme": row["lexeme"],
        }
        state.LEXICON[sub_trait].append(pat)

        pid = len(state.LEXICON_PATTERNS)
        state.LEXICON_PATTERNS.append(pat)
        # Inverted index by token to reduce scanning cost
        for t in tok_set:
            if t:
                state.TOKEN_TO_PATTERN_IDS[t].append(pid)

    ont_emb = torch.load(ONTOLOGY_EMB, map_location="cpu")
    state.ONT_EMBEDDINGS = ont_emb["embeddings"].numpy()
    state.ONT_META = ont_emb["meta"]
    prepare_embedding_index()

    state.tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    encoder = AutoModel.from_pretrained(HF_REPO)
    state.model = OceanModel(encoder, state.LEXICAL_SIZE)

    state_path = hf_hub_download(repo_id=HF_REPO, filename="pytorch_model.bin")
    state_dict = torch.load(state_path, map_location="cpu")
    state.model.load_state_dict(state_dict, strict=False)
    state.model.to(DEVICE)
    state.model.eval()

    n_words, n_phrases = build_fuzzy_index()
    gpu_label = DEVICE_INFO.get("gpu_name") or "CPU"
    logger.info(
        f"Startup OK | device={DEVICE} ({gpu_label}) | "
        f"LEXICAL_SIZE={state.LEXICAL_SIZE} | "
        f"fuzzy_vocab={n_words} phrases={n_phrases}"
    )
