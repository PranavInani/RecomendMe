import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from recommenders.content_based import ContentBasedRecommender
from recommenders.collaborative import CollaborativeRecommender
from recommenders.hybrid import HybridRecommender
from api.schemas import (
    ContentRecommendRequest,
    CollabRecommendRequest,
    HybridRecommendRequest,
    RecommendationResponse,
    MovieSearchResponse,
    UserListResponse,
    UserHistoryResponse
)

content_recommender = None
collab_recommender = None
hybrid_recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global content_recommender, collab_recommender, hybrid_recommender
    print("[Startup] Initializing recommendation engines...")

    data_dir = 'data/ml-latest-small'

    content_recommender = ContentBasedRecommender(data_dir=data_dir)
    content_recommender.fit()

    collab_recommender = CollaborativeRecommender(data_dir=data_dir)
    collab_recommender.movies_df = content_recommender.movies_df
    collab_recommender.fit()

    hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)
    print("[Startup] All recommendation engines online.")
    yield

app = FastAPI(
    title="Hybrid Movie Recommendation Engine",
    description="Full-stack recommendation engine combining Content-Based TF-IDF and SVD Collaborative Filtering.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Recommendation engine is operational."}

@app.get("/api/movies/search", response_model=list[MovieSearchResponse])
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    if not content_recommender:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")
    return content_recommender.search_movies(q, limit=limit)

@app.get("/api/users", response_model=UserListResponse)
def get_users():
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")
    users = collab_recommender.get_users()
    return {"user_ids": users, "total_count": len(users)}

@app.get("/api/users/{user_id}/history", response_model=UserHistoryResponse)
def get_user_history(user_id: int, limit: int = Query(5, ge=1, le=20)):
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")
    history = collab_recommender.get_user_history(user_id, limit=limit)
    return {"user_id": user_id, "history": history}

@app.post("/api/recommend/content", response_model=RecommendationResponse)
def recommend_content(req: ContentRecommendRequest):
    if not content_recommender:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")
    recs = content_recommender.get_recommendations(
        movie_id=req.movie_id,
        movie_title=req.movie_title,
        top_n=req.top_n
    )
    if not recs:
        raise HTTPException(status_code=404, detail="Movie not found or no recommendations available.")
    return {
        "mode": "content-based",
        "seed": req.movie_title or str(req.movie_id),
        "recommendations": recs
    }

@app.post("/api/recommend/collaborative", response_model=RecommendationResponse)
def recommend_collaborative(req: CollabRecommendRequest):
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")
    recs = collab_recommender.recommend_for_user(
        user_id=req.user_id,
        top_n=req.top_n
    )
    return {
        "mode": "collaborative",
        "user_id": req.user_id,
        "recommendations": recs
    }

@app.post("/api/recommend/hybrid", response_model=RecommendationResponse)
def recommend_hybrid(req: HybridRecommendRequest):
    if not hybrid_recommender:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")
    recs = hybrid_recommender.get_recommendations(
        user_id=req.user_id,
        movie_id=req.movie_id,
        movie_title=req.movie_title,
        alpha=req.alpha,
        top_n=req.top_n
    )
    return {
        "mode": "hybrid",
        "user_id": req.user_id,
        "alpha": req.alpha,
        "recommendations": recs
    }

# ── Static frontend (mounted LAST so API routes take priority) ────────────────
DIST_DIR = "frontend/dist"
if os.path.exists(DIST_DIR):
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

    # Catch-all: serve index.html for all non-API routes (SPA routing)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
