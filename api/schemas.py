from pydantic import BaseModel, Field
from typing import List, Optional

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

class MovieSearchResult(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    score: float
    avg_rating: float
    rating_count: int
