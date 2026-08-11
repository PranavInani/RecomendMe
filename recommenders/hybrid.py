from recommenders.content_based import ContentBasedRecommender
from recommenders.collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, cb: ContentBasedRecommender = None, cf: CollaborativeRecommender = None, data_dir: str = 'data/ml-latest-small'):
        self.data_dir = data_dir
        self.cb = cb if cb is not None else ContentBasedRecommender(data_dir=data_dir)
        self.cf = cf if cf is not None else CollaborativeRecommender(data_dir=data_dir)

    def fit(self):
        if self.cb.feature_matrix is None:
            self.cb.fit()
        if self.cf.model is None:
            self.cf.fit()

    def get_recommendations(
        self,
        movie_id: int = None,
        movie_title: str = None,
        user_id: int = None,
        alpha: float = 0.5,  # Weight for content score (0.0 to 1.0)
        top_n: int = 10
    ):
        # 1. Get content recommendations if movie provided
        content_recs = []
        if movie_id is not None or movie_title is not None:
            content_recs = self.cb.get_recommendations(
                movie_id=movie_id,
                movie_title=movie_title,
                top_n=100  # Candidate pool
            )

        # 2. Get collaborative recommendations if user provided
        collab_recs = []
        if user_id is not None:
            collab_recs = self.cf.recommend_for_user(
                user_id=user_id,
                top_n=100  # Candidate pool
            )

        # If only content requested
        if not collab_recs and content_recs:
            return content_recs[:top_n]

        # If only collab requested
        if not content_recs and collab_recs:
            return collab_recs[:top_n]

        # If neither provided, return top rated movies overall
        if not content_recs and not collab_recs:
            return []

        # 3. Combine scores for hybrid mode
        scores_map = {}
        movie_meta = {}

        # Content score dictionary (sim_score is usually 0.0 to 1.0)
        for item in content_recs:
            mid = item['movieId']
            scores_map[mid] = {'content_score': item['similarity_score'], 'collab_score': 0.0}
            movie_meta[mid] = item

        # Collaborative score dictionary (predicted ratings 0.5 to 5.0 -> norm to 0.0 to 1.0)
        for item in collab_recs:
            mid = item['movieId']
            norm_collab = max(0.0, min(1.0, (item['predicted_rating'] - 0.5) / 4.5))
            if mid in scores_map:
                scores_map[mid]['collab_score'] = norm_collab
            else:
                scores_map[mid] = {'content_score': 0.0, 'collab_score': norm_collab}
                movie_meta[mid] = item

        # Calculate hybrid weighted score
        hybrid_results = []
        for mid, scores in scores_map.items():
            c_score = scores['content_score']
            cf_score = scores['collab_score']

            hybrid_score = (alpha * c_score) + ((1.0 - alpha) * cf_score)
            meta = movie_meta[mid]

            hybrid_results.append({
                'movieId': mid,
                'title': meta['title'],
                'genres': meta['genres'],
                'hybrid_score': round(float(hybrid_score), 4),
                'content_score': round(float(c_score), 4),
                'collab_score': round(float(cf_score), 4),
                'avg_rating': meta['avg_rating'],
                'rating_count': meta['rating_count']
            })

        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_results[:top_n]
