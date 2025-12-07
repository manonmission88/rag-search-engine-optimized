from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies,load_stop_words
import string
from nltk.stem import PorterStemmer

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    updated_query = filter_query(preprocess_text(query).split(" "))
    for movie in movies:
        updated_movie = preprocess_text(movie['title'])
        if tokenization_compare(updated_query,updated_movie):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str)-> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text
    
def tokenization_compare(text1: str,text2:  str)->list:
    updated_text1 = text1.split(" ")
    updated_text2 = text2.split(" ")
    for query_text in updated_text1:
        for title_text in updated_text2:
            if query_text in title_text:
                return True 
    return False 
    
    
def filter_query(query):
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    new_query = []
    for word in query:
        if word in stop_words:
            continue 
        new_query.append(word)
    return " ".join([stemmer.stem(q) for q in new_query]) 