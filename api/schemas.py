from pydantic import BaseModel, Field
from typing import List, Optional, Any

# --- Request Schemas ---

class ContentRecommendRequest(BaseModel):
    movie_id: Optional[int] = None
    movie_title: Optional[str] = None
    top_n: int = Field(default=10, ge=1, le=50)

class CollabRecommendRequest(BaseModel):
    user_id: int
    top_n: int = Field(default=10, ge=1, le=50)

class HybridRecommendRequest(BaseModel):
    movie_id: Optional[int] = None
    movie_title: Optional[str] = None
    user_id: Optional[int] = None
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_n: int = Field(default=10, ge=1, le=50)

# --- Response Schemas ---

class MovieSearchResponse(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    score: float
    avg_rating: float
    rating_count: int

class RecommendationItem(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    avg_rating: float = 0.0
    rating_count: int = 0
    similarity_score: Optional[float] = None
    predicted_rating: Optional[float] = None
    hybrid_score: Optional[float] = None
    content_score: Optional[float] = None
    collab_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    mode: str
    seed: Optional[str] = None
    user_id: Optional[int] = None
    alpha: Optional[float] = None
    recommendations: List[RecommendationItem]

class HistoryItem(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    rating: float

class UserHistoryResponse(BaseModel):
    user_id: int
    history: List[HistoryItem]

class UserListResponse(BaseModel):
    user_ids: List[int]
    total_count: int
