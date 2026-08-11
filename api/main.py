import os
import sys
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommenders.content_based import ContentBasedRecommender
from recommenders.collaborative import CollaborativeRecommender
from recommenders.hybrid import HybridRecommender
from api.schemas import (
    ContentRecommendRequest,
    CollabRecommendRequest,
    HybridRecommendRequest,
    MovieSearchResult
)

app = FastAPI(
    title="RecomendMe API",
    description="Hybrid Movie Recommendation Engine (Content-Based + Collaborative SVD)",
    version="1.0.0"
)

# Enable CORS for local development & frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instances
content_recommender = None
collab_recommender = None
hybrid_recommender = None

@app.on_event("startup")
def startup_event():
    global content_recommender, collab_recommender, hybrid_recommender
    print("[API Startup] Initializing models...")
    content_recommender = ContentBasedRecommender()
    content_recommender.fit()

    collab_recommender = CollaborativeRecommender()
    collab_recommender.fit()

    hybrid_recommender = HybridRecommender()
    hybrid_recommender.cb = content_recommender
    hybrid_recommender.cf = collab_recommender
    print("[API Startup] All models initialized successfully!")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "RecomendMe API is online"}

@app.get("/api/movies/search")
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    if not content_recommender:
        raise HTTPException(status_code=503, detail="Models initializing...")
    return content_recommender.search_movies(query=q, limit=limit)

@app.get("/api/users")
def get_users():
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Models initializing...")
    return {
        "user_ids": collab_recommender.user_ids,
        "total_users": len(collab_recommender.user_ids)
    }

@app.get("/api/users/{user_id}/history")
def get_user_history(user_id: int, limit: int = Query(10, ge=1, le=50)):
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Models initializing...")
    history = collab_recommender.get_user_history(user_id=user_id, top_n=limit)
    return {"user_id": user_id, "history": history}

@app.post("/api/recommend/content")
def recommend_content(req: ContentRecommendRequest):
    if not content_recommender:
        raise HTTPException(status_code=503, detail="Models initializing...")

    recs = content_recommender.get_recommendations(
        movie_id=req.movie_id,
        movie_title=req.movie_title,
        top_n=req.top_n
    )
    return {
        "type": "content_based",
        "movie_id": req.movie_id,
        "movie_title": req.movie_title,
        "recommendations": recs
    }

@app.post("/api/recommend/collaborative")
def recommend_collaborative(req: CollabRecommendRequest):
    if not collab_recommender:
        raise HTTPException(status_code=503, detail="Models initializing...")

    recs = collab_recommender.recommend_for_user(
        user_id=req.user_id,
        top_n=req.top_n
    )
    return {
        "type": "collaborative",
        "user_id": req.user_id,
        "recommendations": recs
    }

@app.post("/api/recommend/hybrid")
def recommend_hybrid(req: HybridRecommendRequest):
    if not hybrid_recommender:
        raise HTTPException(status_code=503, detail="Models initializing...")

    recs = hybrid_recommender.get_recommendations(
        movie_id=req.movie_id,
        movie_title=req.movie_title,
        user_id=req.user_id,
        alpha=req.alpha,
        top_n=req.top_n
    )
    return {
        "type": "hybrid",
        "movie_id": req.movie_id,
        "movie_title": req.movie_title,
        "user_id": req.user_id,
        "alpha": req.alpha,
        "recommendations": recs
    }

# Serve frontend static files if built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/dist'))
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="static")

    @app.get("/{full_path:path}")
    def serve_react_app(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
