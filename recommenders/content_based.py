import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
for resource in ['punkt', 'wordnet', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()

import re

# Simple regex-based word splitting for speed
TOKEN_RE = re.compile(r'\w+')

def lemmatize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    tokens = TOKEN_RE.findall(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens])


class ContentBasedRecommender:
    def __init__(self, data_dir: str = 'data/ml-latest-small'):
        self.data_dir = data_dir
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.cosine_sim = None
        self.case_insensitive_titles = []
        self.title_to_idx = {}
        self.idx_to_movie = {}

    def load_data(self):
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        tags_path = os.path.join(self.data_dir, 'tags.csv')
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')

        self.movies_df = pd.read_csv(movies_path)
        if os.path.exists(tags_path):
            self.tags_df = pd.read_csv(tags_path)
        else:
            self.tags_df = pd.DataFrame(columns=['movieId', 'userId', 'tag', 'timestamp'])

        if os.path.exists(ratings_path):
            self.ratings_df = pd.read_csv(ratings_path)
        else:
            self.ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

    def fit(self, save_cache: bool = True, cache_path: str = 'data/content_cache.pkl'):
        if save_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.movies_df = cached_data['movies_df']
                    self.cosine_sim = cached_data['cosine_sim']
                    self._build_indexes()
                    print("[ContentBased] Loaded precomputed model from cache.")
                    return
            except Exception as e:
                print(f"[ContentBased] Failed loading cache ({e}), recomputing...")

        if self.movies_df is None:
            self.load_data()

        print("[ContentBased] Building feature matrix...")
        # One-Hot Encoding for genres
        genres_onehot = self.movies_df['genres'].str.get_dummies()

        # Group tags by movieId
        if not self.tags_df.empty:
            tags_combined = self.tags_df.groupby('movieId')['tag'].apply(
                lambda x: ' '.join([str(t) for t in x if pd.notna(t)])
            ).reset_index()
            df = pd.merge(self.movies_df, tags_combined, on='movieId', how='left')
            df['tag'] = df['tag'].fillna('')
        else:
            df = self.movies_df.copy()
            df['tag'] = ''

        df['text'] = (
            df['title'] + ' ' +
            df['genres'].str.split('|').apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
            df['tag']
        ).str.lower()

        df['text'] = df['text'].apply(lemmatize_text)

        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['text'])

        # Combine TF-IDF matrix with one-hot encoded genres
        combined_features = hstack([tfidf_matrix, genres_onehot])

        # Cosine similarity matrix
        print("[ContentBased] Computing cosine similarity matrix...")
        self.cosine_sim = cosine_similarity(combined_features, combined_features)

        # Merge calculated mean rating into movies_df for rich output
        if not self.ratings_df.empty:
            stats = self.ratings_df.groupby('movieId').agg(
                avg_rating=('rating', 'mean'),
                rating_count=('rating', 'count')
            ).reset_index()
            self.movies_df = pd.merge(self.movies_df, stats, on='movieId', how='left')
            self.movies_df['avg_rating'] = self.movies_df['avg_rating'].round(2).fillna(0.0)
            self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0).astype(int)

        self._build_indexes()

        if save_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'movies_df': self.movies_df,
                    'cosine_sim': self.cosine_sim
                }, f)
            print(f"[ContentBased] Cache saved to {cache_path}")

    def _build_indexes(self):
        self.case_insensitive_titles = [t.lower() for t in self.movies_df['title']]
        self.title_to_idx = {t.lower(): idx for idx, t in enumerate(self.movies_df['title'])}
        self.movieId_to_idx = {row['movieId']: idx for idx, row in self.movies_df.iterrows()}

    def search_movies(self, query: str, limit: int = 10):
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        matches = []
        for idx, row in self.movies_df.iterrows():
            title_lower = row['title'].lower()
            if query_lower in title_lower:
                score = 1.0 if title_lower.startswith(query_lower) else 0.5
                matches.append({
                    'movieId': int(row['movieId']),
                    'title': row['title'],
                    'genres': row['genres'].split('|') if pd.notna(row['genres']) else [],
                    'score': score,
                    'avg_rating': float(row.get('avg_rating', 0)),
                    'rating_count': int(row.get('rating_count', 0))
                })

        matches.sort(key=lambda x: (-x['score'], x['title']))
        return matches[:limit]

    def get_recommendations(self, movie_id: int = None, movie_title: str = None, top_n: int = 10):
        if self.cosine_sim is None:
            self.fit()

        idx = None
        if movie_id is not None and movie_id in self.movieId_to_idx:
            idx = self.movieId_to_idx[movie_id]
        elif movie_title is not None:
            title_lower = movie_title.lower().strip()
            if title_lower in self.title_to_idx:
                idx = self.title_to_idx[title_lower]
            else:
                # Fuzzy fallback
                matches = self.search_movies(movie_title, limit=1)
                if matches:
                    idx = self.movieId_to_idx[matches[0]['movieId']]

        if idx is None:
            return []

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # Exclude self match

        recommendations = []
        for i, score in sim_scores:
            row = self.movies_df.iloc[i]
            recommendations.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'genres': row['genres'].split('|') if pd.notna(row['genres']) else [],
                'similarity_score': round(float(score), 4),
                'avg_rating': float(row.get('avg_rating', 0)),
                'rating_count': int(row.get('rating_count', 0))
            })

        return recommendations
