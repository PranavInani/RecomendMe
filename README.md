# 🎬 RecomendMe — Intelligent Hybrid Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-8.0-purple.svg)](https://vitejs.dev/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)

**RecomendMe** is a full-stack, production-ready hybrid movie recommendation system that combines **Content-Based Filtering (TF-IDF & On-The-Fly Cosine Similarity)** with **Collaborative Filtering (SVD Matrix Factorization)**, served through a high-performance FastAPI backend and a clean, editorial React/Vite web application with Light/Dark mode.

---

## 🌟 Key Features

- **🎯 Content-Based Filtering (Memory-Optimized)**:
  - Analyzes movie titles, lemmatized tags (NLTK), and one-hot encoded genres.
  - Constructs a compact sparse CSR feature matrix (**0.97 MB RAM** vs 724 MB dense matrix).
  - Computes single-row cosine similarity **on-the-fly per request (~1.3 ms)** for sub-10ms response times.

- **👥 Collaborative Filtering (SVD)**:
  - Decomposes the user-item interaction matrix (100,000+ ratings from MovieLens Small) into 50 latent factors using **Singular Value Decomposition (SVD)**.
  - Predicts unobserved movie ratings for any registered user profile.

- **⚡ Dynamic Hybrid Fusion**:
  - Blends content similarity scores and collaborative rating predictions:
    $$\text{Hybrid Score} = \alpha \cdot \text{Content Score} + (1 - \alpha) \cdot \text{Collaborative Score}$$
  - Live interactive weight slider ($\alpha$) in the UI to tune the balance between metadata similarity and community preferences.

- **🎨 Modern Editorial UI & Light/Dark Mode**:
  - Clean, restrained product design (inspired by Linear & Notion) with a flat burnt-orange accent (`#e8590c`).
  - Seamless Light Mode and Dark Mode theme toggle with system preference detection and state persistence.
  - Live search autocomplete with genre tags and average rating stats.

- **☁️ Cloud Free-Tier Ready (Render 512 MB RAM)**:
  - Reduced overall server peak memory usage from **820 MB down to ~53 MB**, making it lightweight and deployable on free cloud tiers.

---

## 🏗️ Architecture & Technology Stack

```
                                  ┌───────────────────────────┐
                                  │      React + Vite UI      │
                                  │  (Editorial + Light/Dark) │
                                  └─────────────┬─────────────┘
                                                │ REST API
                                  ┌─────────────▼─────────────┐
                                  │      FastAPI Backend      │
                                  └──────┬──────────────┬─────┘
                                         │              │
                    ┌────────────────────▼─┐          ┌─▼────────────────────┐
                    │ ContentRecommender   │          │ Collaborative Engine │
                    │(Sparse CSR + Row Sim)│          │ (SVD Factorization)  │
                    └──────────────────────┘          └──────────────────────┘
```

| Component | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-Learn, Scikit-Surprise, NLTK, Pandas, NumPy, SciPy (Sparse CSR) |
| **Frontend** | React, Vite, Lucide React, Vanilla CSS3 (Custom Theme Tokens) |
| **Dataset** | MovieLens Small (`ml-latest-small`) |

---

## 📂 Directory Structure

```
RecomendMe/
├── recommenders/
│   ├── __init__.py
│   ├── content_based.py       # TF-IDF + Lemmatization + Sparse Row Cosine Sim module
│   ├── collaborative.py       # SVD Matrix Factorization module
│   └── hybrid.py              # Dynamic score fusion engine
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI web server & static file host
│   └── schemas.py             # Pydantic request/response schemas
├── data/
│   ├── ml-latest-small/       # MovieLens CSV dataset (movies, ratings, tags)
│   ├── content_cache.pkl      # Lightweight sparse feature cache (~1.5 MB)
│   └── collab_cache.pkl       # Trained SVD model cache (~8 MB)
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React UI component
│   │   ├── App.css            # Component layout styles
│   │   └── index.css          # Design system & Light/Dark tokens
│   ├── dist/                  # Production build directory
│   ├── index.html
│   └── package.json
├── Content_based_Recomendation_system.ipynb  # Original prototype notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Node.js & npm (for building the frontend)

### 2. Setup Environment & Install Dependencies

```bash
# Clone or open project directory
cd /path/to/RecomendMe

# Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python requirements
pip install -r requirements.txt
```

### 3. Build Frontend & Launch Web App

```bash
# Build Vite frontend assets (if modifying frontend)
cd frontend && npm install && npm run build && cd ..

# Launch FastAPI web server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your web browser!

---

## 🔌 API Reference

### 1. Search Movies Autocomplete
- **Endpoint**: `GET /api/movies/search?q={query}&limit={limit}`
- **Description**: Returns live matching movies with metadata.

### 2. Content-Based Recommendations
- **Endpoint**: `POST /api/recommend/content`
- **Body**:
```json
{
  "movie_title": "Toy Story (1995)",
  "top_n": 10
}
```

### 3. Collaborative (SVD) Recommendations
- **Endpoint**: `POST /api/recommend/collaborative`
- **Body**:
```json
{
  "user_id": 1,
  "top_n": 10
}
```

### 4. Hybrid Recommendations
- **Endpoint**: `POST /api/recommend/hybrid`
- **Body**:
```json
{
  "movie_title": "Toy Story (1995)",
  "user_id": 1,
  "alpha": 0.5,
  "top_n": 10
}
```

---

## 💡 How the Algorithms Work

### 1. Content-Based Filtering (Memory-Optimized)
1. **Text Combination**: Concatenates movie title, lemmatized user tags, and genres.
2. **Sparse TF-IDF & One-Hot Encoding**: Converts text features into a sparse Term Frequency-Inverse Document Frequency matrix and merges it with a one-hot encoded genres matrix, stored as a SciPy CSR matrix (`0.97 MB`).
3. **On-the-Fly Cosine Similarity**: Computes similarity scores for the queried movie vector against the sparse feature matrix on-the-fly per request (`1.3 ms` compute time):
   $$\text{Cosine Sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

### 2. SVD Collaborative Filtering
1. **Matrix Factorization**: Decomposes the user-rating matrix $R \approx P \cdot Q^T$, where $P$ is the user latent matrix and $Q$ is the item latent matrix (50 latent factors).
2. **Rating Prediction**: Predicts unseen rating $\hat{r}_{u,i} = \mu + b_u + b_i + p_u^T q_i$.

### 3. Score Normalization & Fusion
1. Content similarity score $S_{\text{content}} \in [0, 1]$.
2. Collaborative rating prediction normalized to $[0, 1]$: $S_{\text{collab}} = \frac{\hat{r}_{u,i} - 0.5}{4.5}$.
3. Hybrid Score = $\alpha \cdot S_{\text{content}} + (1 - \alpha) \cdot S_{\text{collab}}$.

---

## 📜 License
MIT License. Free to use and modify for learning and development.
