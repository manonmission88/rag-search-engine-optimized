import json
import os

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
DATA_PATH1 = os.path.join(PROJECT_ROOT, "data", "stop_words.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list:
    with open(DATA_PATH1, "r") as f:
        data = f.read().splitlines()
    return data
