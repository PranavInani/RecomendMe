import os
import pickle
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD

class CollaborativeRecommender:
    def __init__(self, data_dir: str = 'data/ml-latest-small'):
        self.data_dir = data_dir
        self.movies_df = None
        self.model = None
        self.user_rated_movies = {}  # userId -> set of movieIds rated
        self.user_history_map = {}   # userId -> list of history dicts
        self.all_user_ids = []
        self.all_movie_ids = []

    def load_data(self):
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')

        if self.movies_df is None:
            self.movies_df = pd.read_csv(movies_path)
        
        ratings_df = pd.read_csv(ratings_path)
        return ratings_df

    def fit(self, n_factors: int = 50, n_epochs: int = 20, save_cache: bool = True, cache_path: str = 'data/collab_cache.pkl'):
        if save_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    self.model = cached['model']
                    self.user_rated_movies = cached['user_rated_movies']
                    self.user_history_map = cached['user_history_map']
                    self.all_user_ids = cached['all_user_ids']
                    self.all_movie_ids = cached['all_movie_ids']
                    if 'movies_df' in cached and self.movies_df is None:
                        self.movies_df = cached['movies_df']
                    print("[Collaborative] Loaded trained SVD model from cache.")
                    return
            except Exception as e:
                print(f"[Collaborative] Failed loading cache ({e}), retraining...")

        print("[Collaborative] Training SVD Matrix Factorization model...")
        ratings_df = self.load_data()

        # Compute user rated sets & compact history dict
        self.user_rated_movies = ratings_df.groupby('userId')['movieId'].apply(set).to_dict()
        
        # Compact top history per user
        user_history_map = {}
        merged_history = pd.merge(ratings_df, self.movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
        for user_id, group in merged_history.groupby('userId'):
            top_rated = group.sort_values('rating', ascending=False).head(10)
            user_history_map[int(user_id)] = [
                {
                    'movieId': int(row['movieId']),
                    'title': row['title'],
                    'genres': row['genres'].split('|') if pd.notna(row['genres']) else [],
                    'rating': float(row['rating'])
                }
                for _, row in top_rated.iterrows()
            ]
        self.user_history_map = user_history_map

        self.all_user_ids = sorted(ratings_df['userId'].unique().tolist())
        self.all_movie_ids = sorted(self.movies_df['movieId'].unique().tolist())

        # Fit SVD
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()

        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        self.model.fit(trainset)

        if save_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'user_rated_movies': self.user_rated_movies,
                    'user_history_map': self.user_history_map,
                    'all_user_ids': self.all_user_ids,
                    'all_movie_ids': self.all_movie_ids,
                    'movies_df': self.movies_df
                }, f)
            print(f"[Collaborative] Cache saved to {cache_path}")

    def get_users(self):
        return self.all_user_ids

    def get_user_history(self, user_id: int, limit: int = 5):
        if user_id in self.user_history_map:
            return self.user_history_map[user_id][:limit]
        return []

    def recommend_for_user(self, user_id: int, top_n: int = 10):
        if self.model is None:
            self.fit()

        rated_movies = self.user_rated_movies.get(user_id, set())

        # Unrated movies for this user
        unrated = [m for m in self.all_movie_ids if m not in rated_movies]

        predictions = []
        for movie_id in unrated:
            pred = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_preds = predictions[:top_n]

        # Build output dicts
        movie_dict = self.movies_df.set_index('movieId').to_dict('index')

        recommendations = []
        for movie_id, est_rating in top_preds:
            meta = movie_dict.get(movie_id, {})
            genres = meta.get('genres', '')
            genres_list = genres.split('|') if pd.notna(genres) and genres else []
            recommendations.append({
                'movieId': int(movie_id),
                'title': meta.get('title', f'Movie #{movie_id}'),
                'genres': genres_list,
                'predicted_rating': round(float(est_rating), 2),
                'avg_rating': float(meta.get('avg_rating', 0.0)) if pd.notna(meta.get('avg_rating')) else 0.0,
                'rating_count': int(meta.get('rating_count', 0)) if pd.notna(meta.get('rating_count')) else 0
            })

        return recommendations
