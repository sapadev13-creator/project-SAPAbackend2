import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
ONTOLOGY_CSV = BASE_DIR / "ontology_clean.csv"
ONTOLOGY_EMB = BASE_DIR / "ontology_embeddings.pt"
KEYWORDS_XLSX = BASE_DIR / "keywords_traits.xlsx"

HF_REPO = "sapadev13/sapa_ocean_id"
DEVICE = "cpu"
MAX_LEN = 256

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")
TWITTER_REDIRECT_URI = "http://localhost:8000/auth/twitter/callback"
TWITTER_SCOPES = ["tweet.read", "users.read", "offline.access"]
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-secret-CHANGE-ME")
FRONTEND_URL = "http://localhost:3000"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"

OCEAN_COLORS = {
    "O": "#6366F1",
    "C": "#22C55E",
    "E": "#F59E0B",
    "A": "#3B82F6",
    "N": "#EF4444",
}

OCEAN_TRAITS = ["O", "C", "E", "A", "N"]

# Semantic ontology (cosine similarity vs ontology_embeddings.pt)
SEMANTIC_TOP_K = 10
SEMANTIC_THRESHOLD = 0.50
SEMANTIC_LEXICAL_WEIGHT = 1.2
SEMANTIC_OCEAN_WEIGHT = 0.18
SEMANTIC_PHRASE_BOOST = 0.92

# Fuzzy typo correction
FUZZY_MIN_RATIO = 0.84
FUZZY_PHRASE_MIN_RATIO = 0.88
FUZZY_SHORT_MIN_RATIO = 0.80

SUBTRAIT_OCEAN_PREFIX = {
    "c_": "C",
    "o_": "O",
    "e_": "E",
    "a_": "A",
    "n_": "N",
}
OCEAN_LABELS = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

TRAIT_LIST_NAMES = [
    "NEGATIVE_SOCIAL",
    "POSITIVE_SOCIAL",
    "EMO_POSITIVE",
    "EMO_NEGATIVE",
    "INTROSPECTION",
    "ACHIEVEMENT",
    "CREATIVE_DISCUSSION_A",
    "TRUST",
    "RELATIONSHIP_AFFECTION",
    "COLLABORATION",
    "ANGER_EMO",
    "SAD_EMO",
    "ANXIETY_EMO",
    "EXTREME_NEGATIVE",
    "DISCIPLINE_C",
    "EXTRAVERSION_E",
    "E_SOCIAL_DEPENDENCY",
    "EMPATHY_HARMONY_A",
    "MENTAL_UNSTABLE_N",
]
