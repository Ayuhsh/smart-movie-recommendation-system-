import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

def extract_names(obj):
    try:
        return ' '.join([i['name'] for i in ast.literal_eval(obj)])
    except:
        return ''

movies['genres_str'] = movies['genres'].apply(extract_names)
movies['keywords_str'] = movies['keywords'].apply(extract_names)
movies['combined'] = (
    movies['overview'].fillna('') + ' ' +
    movies['genres_str'] + ' ' +
    movies['keywords_str']
)

movies = movies[['movie_id', 'title', 'combined']].dropna()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

with open('movie_data.pkl', 'wb') as f:
    pickle.dump((movies, cosine_sim), f)

print("✅ movie_data.pkl ban gayi!")
