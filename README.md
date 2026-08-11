# 🎬 RecomendMe — Intelligent Hybrid Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-8.0-purple.svg)](https://vitejs.dev/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)

**RecomendMe** is a full-stack, state-of-the-art hybrid recommendation system that combines **Content-Based Filtering (TF-IDF & Cosine Similarity)** with **Collaborative Filtering (SVD Matrix Factorization)** and serves them through a high-performance FastAPI backend and a modern React/Vite web application.

---

## 🌟 Key Features

- **🎯 Content-Based Filtering**:
  - Analyzes movie titles, lemmatized tags (NLTK), and one-hot encoded genres.
  - Constructs sparse feature matrices and computes cosine similarity for instant item-item recommendations.
  - Precomputed disk caching (`data/content_cache.pkl`) for sub-100ms response times.

- **👥 Collaborative Filtering (SVD)**:
  - Decomposes the user-item interaction matrix (100,000+ ratings from MovieLens Small) into 50 latent factors using **Singular Value Decomposition (SVD)**.
  - Predicts unobserved movie ratings for any registered user profile.

- **⚡ Dynamic Hybrid Fusion**:
  - Blends content similarity scores and collaborative rating predictions:
    $$\text{Hybrid Score} = \alpha \cdot \text{Content Score} + (1 - \alpha) \cdot \text{Collaborative Score}$$
  - Live interactive weight slider ($\alpha$) in the UI to tune the balance between metadata similarity and community preferences.

- **✨ Glassmorphism UI & Web Application**:
  - Live search autocomplete with genre pills and rating stats as you type.
  - User profile selector with user history previews.
  - Responsive, dark-themed glassmorphism interface built with React & Vite.

---

## 🏗️ Architecture & Technology Stack

```
                                  ┌───────────────────────────┐
                                  │      React + Vite UI      │
                                  │   (Glassmorphism Frontend)│
                                  └─────────────┬─────────────┘
                                                │ REST API
                                  ┌─────────────▼─────────────┐
                                  │      FastAPI Backend      │
                                  └──────┬──────────────┬─────┘
                                         │              │
                    ┌────────────────────▼─┐          ┌─▼────────────────────┐
                    │ ContentRecommender   │          │ Collaborative Engine │
                    │ (TF-IDF + Cosine Sim)│          │ (SVD Factorization)  │
                    └──────────────────────┘          └──────────────────────┘
```

| Component | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-Learn, Scikit-Surprise, NLTK, Pandas, NumPy, SciPy |
| **Frontend** | React, Vite, Lucide React, Vanilla CSS3 (Glassmorphism) |
| **Dataset** | MovieLens Small (`ml-latest-small`) |

---

## 📂 Directory Structure

```
RecomendMe/
├── recommenders/
│   ├── __init__.py
│   ├── content_based.py       # TF-IDF + Lemmatization + Cosine Sim module
│   ├── collaborative.py       # SVD Matrix Factorization module
│   └── hybrid.py              # Dynamic score fusion engine
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI web server & static file host
│   └── schemas.py             # Pydantic request/response schemas
├── data/
│   ├── ml-latest-small/       # MovieLens CSV dataset (movies, ratings, tags)
│   ├── content_cache.pkl      # Pre-calculated content similarity cache
│   └── collab_cache.pkl       # Trained SVD model cache
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React UI component
│   │   ├── App.css            # Component styles
│   │   └── index.css          # Design system & tokens
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

### 1. Content-Based Filtering
1. **Text Combination**: Concatenates movie title, lemmatized user tags, and genres.
2. **TF-IDF & One-Hot Encoding**: Converts text features into a sparse Term Frequency-Inverse Document Frequency matrix and merges it with a one-hot encoded genres matrix.
3. **Cosine Similarity**: Computes similarity scores between vectors:
   $$\text{Cosine Sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

### 2. SVD Collaborative Filtering
1. **Matrix Factorization**: Decomposes the user-rating matrix $R \approx P \cdot Q^T$, where $P$ is the user latent matrix and $Q$ is the item latent matrix.
2. **Rating Prediction**: Predicts unseen rating $\hat{r}_{u,i} = \mu + b_u + b_i + p_u^T q_i$.

### 3. Score Normalization & Fusion
1. Content similarity score $S_{\text{content}} \in [0, 1]$.
2. Collaborative rating prediction normalized to $[0, 1]$: $S_{\text{collab}} = \frac{\hat{r}_{u,i} - 0.5}{4.5}$.
3. Hybrid Score = $\alpha \cdot S_{\text{content}} + (1 - \alpha) \cdot S_{\text{collab}}$.

---

## 📜 License
MIT License. Free to use and modify for learning and development.
