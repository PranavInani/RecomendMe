import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Film, 
  Users, 
  Sliders, 
  Search, 
  Star, 
  BookOpen, 
  Layers, 
  CheckCircle2, 
  Loader2, 
  TrendingUp,
  Cpu,
  Zap,
  Info
} from 'lucide-react';
import './App.css';

const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('hybrid'); // 'content', 'collaborative', 'hybrid', 'architecture'
  
  // Search & Inputs
  const [searchQuery, setSearchQuery] = useState('Toy Story (1995)');
  const [selectedMovie, setSelectedMovie] = useState({ movieId: 1, title: 'Toy Story (1995)' });
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState(1);
  const [userHistory, setUserHistory] = useState([]);
  
  const [alpha, setAlpha] = useState(0.5); // 0.0 = full collab, 1.0 = full content
  const [topN, setTopN] = useState(8);
  
  // Results & Loading
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch initial user IDs
  useEffect(() => {
    fetch(`${API_BASE}/users`)
      .then(res => res.json())
      .then(data => {
        if (data.user_ids && data.user_ids.length > 0) {
          setUsers(data.user_ids.slice(0, 50)); // top 50 users for dropdown
        }
      })
      .catch(err => console.error("Error fetching users:", err));
  }, []);

  // Fetch user history when selectedUserId changes
  useEffect(() => {
    if (selectedUserId) {
      fetch(`${API_BASE}/users/${selectedUserId}/history?limit=5`)
        .then(res => res.json())
        .then(data => setUserHistory(data.history || []))
        .catch(err => console.error("Error fetching history:", err));
    }
  }, [selectedUserId]);

  // Live search suggestions
  useEffect(() => {
    if (searchQuery.trim().length > 1 && showSuggestions) {
      const timer = setTimeout(() => {
        fetch(`${API_BASE}/movies/search?q=${encodeURIComponent(searchQuery)}&limit=6`)
          .then(res => res.json())
          .then(data => setSuggestions(data))
          .catch(err => console.error(err));
      }, 200);
      return () => clearTimeout(timer);
    } else {
      setSuggestions([]);
    }
  }, [searchQuery, showSuggestions]);

  // Handle Recommendation Request
  const handleGetRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      let endpoint = `${API_BASE}/recommend/hybrid`;
      let body = {};

      if (activeTab === 'content') {
        endpoint = `${API_BASE}/recommend/content`;
        body = {
          movie_title: selectedMovie ? selectedMovie.title : searchQuery,
          movie_id: selectedMovie ? selectedMovie.movieId : null,
          top_n: topN
        };
      } else if (activeTab === 'collaborative') {
        endpoint = `${API_BASE}/recommend/collaborative`;
        body = {
          user_id: Number(selectedUserId),
          top_n: topN
        };
      } else {
        // Hybrid mode
        endpoint = `${API_BASE}/recommend/hybrid`;
        body = {
          movie_title: selectedMovie ? selectedMovie.title : searchQuery,
          movie_id: selectedMovie ? selectedMovie.movieId : null,
          user_id: Number(selectedUserId),
          alpha: parseFloat(alpha),
          top_n: topN
        };
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch recommendations. Ensure API backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Trigger recommendations on tab load or initial render
  useEffect(() => {
    handleGetRecommendations();
  }, [activeTab]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header-bar">
        <div className="logo-group">
          <div className="logo-icon">
            <Sparkles size={24} />
          </div>
          <div>
            <h1 className="logo-title">
              Recomend<span className="gradient-text">Me</span>
            </h1>
            <p className="logo-subtitle">Next-Gen Hybrid Recommendation Engine</p>
          </div>
        </div>

        <div className="status-pill">
          <div className="status-dot"></div>
          <span>SVD + TF-IDF Hybrid Engine Online</span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'content' ? 'active' : ''}`}
          onClick={() => setActiveTab('content')}
        >
          <Film size={18} />
          <span>Content-Based</span>
        </button>

        <button 
          className={`tab-btn ${activeTab === 'collaborative' ? 'active' : ''}`}
          onClick={() => setActiveTab('collaborative')}
        >
          <Users size={18} />
          <span>Collaborative (SVD)</span>
        </button>

        <button 
          className={`tab-btn ${activeTab === 'hybrid' ? 'active' : ''}`}
          onClick={() => setActiveTab('hybrid')}
        >
          <Zap size={18} />
          <span>Hybrid Fusion</span>
        </button>

        <button 
          className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          <BookOpen size={18} />
          <span>How It Works</span>
        </button>
      </nav>

      {/* Main Content Grid */}
      {activeTab !== 'architecture' ? (
        <main className="main-content">
          {/* Controls Side Panel */}
          <aside className="glass-panel control-panel">
            <h2 className="panel-title">
              <Sliders size={20} className="gradient-text" />
              <span>Config Engine</span>
            </h2>

            {/* Movie Title Input (for Content & Hybrid) */}
            {(activeTab === 'content' || activeTab === 'hybrid') && (
              <div className="form-group">
                <label className="form-label">
                  <span>Seed Movie Title</span>
                  <span className="badge badge-purple">TF-IDF</span>
                </label>
                <div className="input-wrapper">
                  <Search size={18} className="input-icon" />
                  <input
                    type="text"
                    className="text-input"
                    placeholder="Search movie title..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      setShowSuggestions(true);
                    }}
                    onFocus={() => setShowSuggestions(true)}
                  />
                </div>

                {/* Autocomplete Suggestions */}
                {showSuggestions && suggestions.length > 0 && (
                  <div className="suggestions-list">
                    {suggestions.map((movie) => (
                      <div
                        key={movie.movieId}
                        className="suggestion-item"
                        onClick={() => {
                          setSelectedMovie(movie);
                          setSearchQuery(movie.title);
                          setShowSuggestions(false);
                        }}
                      >
                        <div className="suggestion-title">{movie.title}</div>
                        <div className="suggestion-meta">
                          <span>{movie.genres.slice(0, 3).join(', ')}</span>
                          <span>★ {movie.avg_rating}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* User ID Select (for Collaborative & Hybrid) */}
            {(activeTab === 'collaborative' || activeTab === 'hybrid') && (
              <div className="form-group">
                <label className="form-label">
                  <span>Select User Profile</span>
                  <span className="badge badge-cyan">SVD Factors</span>
                </label>
                <div className="input-wrapper">
                  <Users size={18} className="input-icon" />
                  <select
                    className="select-input"
                    value={selectedUserId}
                    onChange={(e) => setSelectedUserId(Number(e.target.value))}
                  >
                    {users.map(uid => (
                      <option key={uid} value={uid}>User #{uid}</option>
                    ))}
                  </select>
                </div>

                {/* User History Teaser */}
                {userHistory.length > 0 && (
                  <div style={{ marginTop: '6px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>User #{selectedUserId} Loved: </span>
                    {userHistory.map(h => h.title.split(' (')[0]).slice(0, 2).join(', ')}
                  </div>
                )}
              </div>
            )}

            {/* Hybrid Alpha Weight Slider */}
            {activeTab === 'hybrid' && (
              <div className="slider-container">
                <div className="form-label">
                  <span>Hybrid Weight Balance (α)</span>
                  <span className="badge badge-amber">{Math.round(alpha * 100)}% Content / {Math.round((1 - alpha) * 100)}% Collab</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  className="range-slider"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                />
                <div className="slider-labels">
                  <span>← 100% SVD Rating</span>
                  <span>100% TF-IDF Text →</span>
                </div>
              </div>
            )}

            {/* Number of Recommendations */}
            <div className="form-group">
              <label className="form-label">Number of Results</label>
              <select
                className="select-input"
                style={{ paddingLeft: '14px' }}
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
              >
                <option value={5}>Top 5 Movies</option>
                <option value={8}>Top 8 Movies</option>
                <option value={12}>Top 12 Movies</option>
                <option value={20}>Top 20 Movies</option>
              </select>
            </div>

            {/* Generate Button */}
            <button 
              className="btn-primary" 
              onClick={handleGetRecommendations}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="spin" />
                  <span>Computing Vectors...</span>
                </>
              ) : (
                <>
                  <Zap size={18} />
                  <span>Generate Recommendations</span>
                </>
              )}
            </button>
          </aside>

          {/* Results Grid Panel */}
          <section className="results-panel">
            <div className="glass-panel results-header">
              <div className="results-title-group">
                <TrendingUp size={22} className="gradient-text" />
                <div>
                  <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
                    {activeTab === 'content' && `Similar to "${selectedMovie ? selectedMovie.title : searchQuery}"`}
                    {activeTab === 'collaborative' && `Personalized Predictions for User #${selectedUserId}`}
                    {activeTab === 'hybrid' && `Hybrid Blended Results for User #${selectedUserId}`}
                  </h2>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Top ranked items generated in real-time
                  </p>
                </div>
              </div>
              <span className="badge badge-purple">
                {recommendations.length} Results
              </span>
            </div>

            {error && (
              <div className="glass-panel" style={{ padding: '20px', color: '#f87171', border: '1px solid rgba(248, 113, 113, 0.3)' }}>
                {error}
              </div>
            )}

            {/* Movie Grid */}
            {recommendations.length > 0 ? (
              <div className="movie-grid">
                {recommendations.map((item, idx) => (
                  <article key={item.movieId || idx} className="glass-panel glass-panel-hover movie-card">
                    <div className="card-header">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className="badge badge-cyan">#{idx + 1}</span>
                        
                        {item.similarity_score && (
                          <span className="score-badge" style={{ color: '#a78bfa' }}>
                            {Math.round(item.similarity_score * 100)}% Sim
                          </span>
                        )}

                        {item.predicted_rating && (
                          <span className="score-badge" style={{ color: '#38bdf8' }}>
                            ★ {item.predicted_rating.toFixed(2)} Pred
                          </span>
                        )}

                        {item.hybrid_score && (
                          <span className="score-badge" style={{ color: '#34d399' }}>
                            {(item.hybrid_score * 100).toFixed(1)} Hybrid
                          </span>
                        )}
                      </div>

                      <h3 className="card-title">{item.title}</h3>

                      <div className="genres-wrapper">
                        {item.genres && item.genres.map((g, i) => (
                          <span key={i} className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                            {g}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="card-metrics">
                      <div className="rating-info">
                        <Star size={15} fill="currentColor" />
                        <span>{item.avg_rating ? item.avg_rating.toFixed(1) : 'N/A'}</span>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                          ({item.rating_count || 0})
                        </span>
                      </div>

                      {item.content_score !== undefined && item.collab_score !== undefined && (
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          Txt: {Math.round(item.content_score * 100)}% | SVD: {Math.round(item.collab_score * 100)}%
                        </div>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              !loading && (
                <div className="glass-panel empty-state">
                  <div className="empty-icon">
                    <Film size={32} />
                  </div>
                  <h3>No Recommendations Yet</h3>
                  <p>Select a seed movie or user profile on the left and click Generate!</p>
                </div>
              )
            )}
          </section>
        </main>
      ) : (
        /* Explainer & Architecture View */
        <section className="explainer-grid">
          <article className="glass-panel explainer-card">
            <div className="explainer-icon" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#c084fc' }}>
              <Film size={24} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>1. Content-Based Filtering</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              Uses TF-IDF Vectorization on title, lemmatized tags, and genre matrices to construct movie feature vectors. Calculates Cosine Similarity between vectors to find items with similar metadata.
            </p>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="#34d399" /> Cold-start friendly for new movies
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="#34d399" /> Analyzes genres & NLTK lemmatized tags
              </li>
            </ul>
          </article>

          <article className="glass-panel explainer-card">
            <div className="explainer-icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#38bdf8' }}>
              <Users size={24} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>2. SVD Collaborative Filtering</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              Decomposes the 100,000+ ratings user-item interaction matrix into low-rank latent feature vectors (50 factors) using Singular Value Decomposition (SVD). Predicts unobserved rating preferences.
            </p>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="#34d399" /> Captures implicit user preference patterns
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="#34d399" /> Matrix factorization via Surprise framework
              </li>
            </ul>
          </article>

          <article className="glass-panel explainer-card">
            <div className="explainer-icon" style={{ background: 'rgba(236, 72, 153, 0.15)', color: '#f472b6' }}>
              <Zap size={24} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>3. Dynamic Hybrid Fusion</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              Combines normalized scores from both content-based similarity and SVD collaborative rating predictions: <code>score = α·Content + (1-α)·Collab</code>.
            </p>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="#34d399" /> Live weight tuning via interactive slider
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="#34d399" /> Best of both worlds accuracy & diversity
              </li>
            </ul>
          </article>
        </section>
      )}
    </div>
  );
}
