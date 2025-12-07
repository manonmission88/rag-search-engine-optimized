from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        updated_query = preprocess_text(query)
        updated_movie = preprocess_text(movie['title'])
        if updated_query in updated_movie:
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str)-> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text
    
    