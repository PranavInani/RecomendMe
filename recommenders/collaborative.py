import os
import pickle
import pandas as pd
import numpy as np

try:
    from surprise import Dataset, Reader, SVD
    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False

class CollaborativeRecommender:
    def __init__(self, data_dir: str = 'data/ml-latest-small'):
        self.data_dir = data_dir
        self.ratings_df = None
        self.movies_df = None
        self.svd_model = None
        self.user_ids = []
        self.movie_ids = []
        self.movie_dict = {}

    def load_data(self):
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')

        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)

        # Build movie dict for fast metadata lookup
        for _, row in self.movies_df.iterrows():
            self.movie_dict[int(row['movieId'])] = {
                'title': row['title'],
                'genres': row['genres'].split('|') if pd.notna(row['genres']) else []
            }

        # Calculate average ratings
        stats = self.ratings_df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).to_dict('index')

        for mid, data in self.movie_dict.items():
            st = stats.get(mid, {'avg_rating': 0.0, 'rating_count': 0})
            data['avg_rating'] = round(float(st['avg_rating']), 2)
            data['rating_count'] = int(st['rating_count'])

        self.user_ids = sorted(self.ratings_df['userId'].unique().tolist())
        self.movie_ids = sorted(self.movies_df['movieId'].unique().tolist())

    def fit(self, save_cache: bool = True, cache_path: str = 'data/collab_cache.pkl'):
        if save_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.svd_model = cached_data['svd_model']
                    self.movies_df = cached_data['movies_df']
                    self.ratings_df = cached_data['ratings_df']
                    self.movie_dict = cached_data['movie_dict']
                    self.user_ids = cached_data['user_ids']
                    self.movie_ids = cached_data['movie_ids']
                    print("[Collaborative] Loaded precomputed SVD model from cache.")
                    return
            except Exception as e:
                print(f"[Collaborative] Failed loading cache ({e}), recomputing...")

        if self.ratings_df is None:
            self.load_data()

        print("[Collaborative] Training SVD Matrix Factorization model...")

        if HAS_SURPRISE:
            reader = Reader(rating_scale=(0.5, 5.0))
            data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)
            trainset = data.build_full_trainset()

            # Train SVD with 50 factors
            self.svd_model = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=42)
            self.svd_model.fit(trainset)
        else:
            print("[Collaborative] scikit-surprise not found, skipping SVD.")

        if save_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'svd_model': self.svd_model,
                    'movies_df': self.movies_df,
                    'ratings_df': self.ratings_df,
                    'movie_dict': self.movie_dict,
                    'user_ids': self.user_ids,
                    'movie_ids': self.movie_ids
                }, f)
            print(f"[Collaborative] Cache saved to {cache_path}")

    def get_user_history(self, user_id: int, top_n: int = 10):
        if self.ratings_df is None:
            self.load_data()

        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if user_ratings.empty:
            return []

        top_user_ratings = user_ratings.sort_values(by=['rating', 'timestamp'], ascending=[False, False]).head(top_n)

        history = []
        for _, row in top_user_ratings.iterrows():
            mid = int(row['movieId'])
            meta = self.movie_dict.get(mid, {'title': f'Movie {mid}', 'genres': [], 'avg_rating': 0, 'rating_count': 0})
            history.append({
                'movieId': mid,
                'title': meta['title'],
                'genres': meta['genres'],
                'user_rating': float(row['rating']),
                'avg_rating': meta['avg_rating'],
                'rating_count': meta['rating_count']
            })
        return history

    def recommend_for_user(self, user_id: int, top_n: int = 10):
        if self.svd_model is None:
            self.fit()

        if user_id not in self.user_ids:
            # Fallback to overall top rated movies for unknown users
            top_movies = self.movies_df.sort_values(by=['avg_rating', 'rating_count'], ascending=[False, False]).head(top_n)
            results = []
            for _, row in top_movies.iterrows():
                results.append({
                    'movieId': int(row['movieId']),
                    'title': row['title'],
                    'genres': row['genres'].split('|') if pd.notna(row['genres']) else [],
                    'predicted_rating': 4.0,
                    'avg_rating': float(row.get('avg_rating', 0)),
                    'rating_count': int(row.get('rating_count', 0))
                })
            return results

        # Get movies user has already rated
        rated_movie_ids = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])

        # Predict ratings for all unrated movies
        predictions = []
        for mid in self.movie_ids:
            if mid not in rated_movie_ids:
                pred = self.svd_model.predict(user_id, mid)
                predictions.append((mid, pred.est))

        # Sort predictions descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]

        recommendations = []
        for mid, est_rating in top_predictions:
            meta = self.movie_dict.get(mid, {'title': f'Movie {mid}', 'genres': [], 'avg_rating': 0, 'rating_count': 0})
            recommendations.append({
                'movieId': mid,
                'title': meta['title'],
                'genres': meta['genres'],
                'predicted_rating': round(float(est_rating), 2),
                'avg_rating': meta['avg_rating'],
                'rating_count': meta['rating_count']
            })

        return recommendations
