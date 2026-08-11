import os
import pickle
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
for resource in ['punkt', 'wordnet', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()
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
        self.feature_matrix = None  # Sparse CSR matrix (0.97 MB RAM vs 724 MB dense cosine sim)
        self.case_insensitive_titles = []
        self.title_to_idx = {}
        self.movieId_to_idx = {}

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
                    self.feature_matrix = cached_data['feature_matrix']
                    self._build_indexes()
                    print("[ContentBased] Loaded precomputed sparse feature model from cache (~2 MB RAM).")
                    return
            except Exception as e:
                print(f"[ContentBased] Failed loading cache ({e}), recomputing...")

        if self.movies_df is None:
            self.load_data()

        print("[ContentBased] Building sparse feature matrix...")
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

        # Combine TF-IDF matrix with one-hot encoded genres into CSR sparse matrix
        combined_features = hstack([tfidf_matrix, genres_onehot]).tocsr()
        self.feature_matrix = combined_features

        # Merge calculated mean rating into movies_df for rich output
        if self.ratings_df is not None and not self.ratings_df.empty:
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
                    'feature_matrix': self.feature_matrix
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
        if self.feature_matrix is None:
            self.fit()

        idx = None
        if movie_id is not None and movie_id in self.movieId_to_idx:
            idx = self.movieId_to_idx[movie_id]
        elif movie_title is not None:
            title_lower = movie_title.lower().strip()
            if title_lower in self.title_to_idx:
                idx = self.title_to_idx[title_lower]
            else:
                matches = self.search_movies(movie_title, limit=1)
                if matches:
                    idx = self.movieId_to_idx[matches[0]['movieId']]

        if idx is None:
            return []

        # ON-THE-FLY: Compute cosine similarity for single row vs all rows (1.3 ms, ~76 KB)
        sim_scores_row = cosine_similarity(self.feature_matrix[idx:idx+1], self.feature_matrix)[0]

        # Top-N indices descending (exclude self)
        top_indices = np.argsort(sim_scores_row)[::-1][1:top_n + 1]

        recommendations = []
        for i in top_indices:
            score = sim_scores_row[i]
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
