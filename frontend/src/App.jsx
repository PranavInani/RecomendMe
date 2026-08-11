import React, { useState, useEffect } from 'react';
import { 
  Film, 
  Users, 
  Sliders, 
  Search, 
  Star, 
  BookOpen, 
  CheckCircle2, 
  Loader2, 
  Zap,
  Sun,
  Moon
} from 'lucide-react';
import './App.css';

const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('hybrid'); // 'content', 'collaborative', 'hybrid', 'architecture'
  
  // Theme state (light / dark)
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') || 
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    }
    return 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

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
          setUsers(data.user_ids.slice(0, 50));
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

  // Trigger recommendations on tab load
  useEffect(() => {
    handleGetRecommendations();
  }, [activeTab]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header-bar">
        <div className="logo-group">
          <div className="logo-icon">
            <Film size={20} />
          </div>
          <div>
            <h1 className="logo-title">RecomendMe</h1>
            <p className="logo-subtitle">Movie Recommendations</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="status-pill">
            <div className="status-dot"></div>
            <span>System Online</span>
          </div>

          <button 
            className="theme-toggle" 
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'content' ? 'active' : ''}`}
          onClick={() => setActiveTab('content')}
        >
          <Film size={16} />
          <span>Content-Based</span>
        </button>

        <button 
          className={`tab-btn ${activeTab === 'collaborative' ? 'active' : ''}`}
          onClick={() => setActiveTab('collaborative')}
        >
          <Users size={16} />
          <span>Collaborative</span>
        </button>

        <button 
          className={`tab-btn ${activeTab === 'hybrid' ? 'active' : ''}`}
          onClick={() => setActiveTab('hybrid')}
        >
          <Zap size={16} />
          <span>Hybrid Fusion</span>
        </button>

        <button 
          className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          <BookOpen size={16} />
          <span>How It Works</span>
        </button>
      </nav>

      {/* Main Content Grid */}
      {activeTab !== 'architecture' ? (
        <main className="main-content">
          {/* Controls Side Panel */}
          <aside className="card control-panel">
            <h2 className="panel-title">
              <Sliders size={16} />
              <span>Settings</span>
            </h2>

            {/* Movie Title Input */}
            {(activeTab === 'content' || activeTab === 'hybrid') && (
              <div className="form-group">
                <label className="form-label">
                  <span>Movie</span>
                  <span className="tag">TF-IDF</span>
                </label>
                <div className="input-wrapper">
                  <Search size={16} className="input-icon" />
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

            {/* User ID Select */}
            {(activeTab === 'collaborative' || activeTab === 'hybrid') && (
              <div className="form-group">
                <label className="form-label">
                  <span>User Profile</span>
                  <span className="tag">SVD</span>
                </label>
                <div className="input-wrapper">
                  <Users size={16} className="input-icon" />
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
                  <div style={{ marginTop: '4px', fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
                    <span style={{ fontWeight: 500, color: 'var(--text-secondary)' }}>Loved: </span>
                    {userHistory.map(h => h.title.split(' (')[0]).slice(0, 2).join(', ')}
                  </div>
                )}
              </div>
            )}

            {/* Hybrid Weight Slider */}
            {activeTab === 'hybrid' && (
              <div className="slider-container">
                <div className="form-label">
                  <span>Content vs Collaborative</span>
                  <span className="tag tag--accent">{Math.round(alpha * 100)}% / {Math.round((1 - alpha) * 100)}%</span>
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
                  <span>Collaborative (SVD)</span>
                  <span>Content (Text)</span>
                </div>
              </div>
            )}

            {/* Number of Recommendations */}
            <div className="form-group">
              <label className="form-label">Number of Results</label>
              <select
                className="select-input"
                style={{ paddingLeft: '12px' }}
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
                  <Loader2 size={16} className="spin" />
                  <span>Loading...</span>
                </>
              ) : (
                <span>Get Recommendations</span>
              )}
            </button>
          </aside>

          {/* Results Grid Panel */}
          <section className="results-panel">
            <div className="card results-header">
              <div className="results-title-group">
                <h2 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {activeTab === 'content' && `Similar to "${selectedMovie ? selectedMovie.title : searchQuery}"`}
                  {activeTab === 'collaborative' && `Recommended for User #${selectedUserId}`}
                  {activeTab === 'hybrid' && `Hybrid Recommendations for User #${selectedUserId}`}
                </h2>
              </div>
              <span className="tag">
                {recommendations.length} Results
              </span>
            </div>

            {error && (
              <div className="card" style={{ padding: '16px', color: 'var(--error)', border: '1px solid var(--error)' }}>
                {error}
              </div>
            )}

            {/* Movie Grid */}
            {recommendations.length > 0 ? (
              <div className="movie-grid">
                {recommendations.map((item, idx) => (
                  <article key={item.movieId || idx} className="card movie-card">
                    <div className="card-header">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className="tag">#{idx + 1}</span>
                        
                        {item.similarity_score !== undefined && (
                          <span className="score-badge">
                            {Math.round(item.similarity_score * 100)}% match
                          </span>
                        )}

                        {item.predicted_rating !== undefined && (
                          <span className="score-badge">
                            ★ {item.predicted_rating.toFixed(1)} pred
                          </span>
                        )}

                        {item.hybrid_score !== undefined && (
                          <span className="score-badge">
                            {Math.round(item.hybrid_score * 100)}% match
                          </span>
                        )}
                      </div>

                      <h3 className="card-title">{item.title}</h3>

                      <div className="genres-wrapper">
                        {item.genres && item.genres.map((g, i) => (
                          <span key={i} className="tag" style={{ fontSize: '0.7rem' }}>
                            {g}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="card-metrics">
                      <div className="rating-info">
                        <Star size={14} fill="currentColor" />
                        <span>{item.avg_rating ? item.avg_rating.toFixed(1) : 'N/A'}</span>
                        <span style={{ color: 'var(--text-tertiary)', fontSize: '0.75rem' }}>
                          ({item.rating_count || 0})
                        </span>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              !loading && (
                <div className="card empty-state">
                  <div className="empty-icon">
                    <Film size={24} />
                  </div>
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>No Recommendations Yet</h3>
                  <p style={{ fontSize: '0.8125rem' }}>Select a movie or user profile on the left and click Get Recommendations.</p>
                </div>
              )
            )}
          </section>
        </main>
      ) : (
        /* Explainer & Architecture View */
        <section className="explainer-grid">
          <article className="card explainer-card">
            <div className="explainer-icon">
              <Film size={20} />
            </div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>1. Content-Based Filtering</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              Uses TF-IDF Vectorization on movie titles, lemmatized tags, and genre matrices. Calculates Cosine Similarity between item feature vectors.
            </p>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} color="var(--success)" /> Cold-start friendly for new movies
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} color="var(--success)" /> NLTK text processing
              </li>
            </ul>
          </article>

          <article className="card explainer-card">
            <div className="explainer-icon">
              <Users size={20} />
            </div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>2. SVD Collaborative Filtering</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              Decomposes the 100,000+ ratings user-item interaction matrix into 50 latent factors using Singular Value Decomposition (SVD).
            </p>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} color="var(--success)" /> Captures implicit user preferences
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} color="var(--success)" /> Matrix factorization via Surprise
              </li>
            </ul>
          </article>

          <article className="card explainer-card">
            <div className="explainer-icon">
              <Zap size={20} />
            </div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>3. Dynamic Hybrid Fusion</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              Combines normalized scores from both content-based similarity and SVD rating predictions: <code>score = α·Content + (1-α)·Collab</code>.
            </p>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} color="var(--success)" /> Live weight tuning via slider
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} color="var(--success)" /> Blends text similarity & rating trends
              </li>
            </ul>
          </article>
        </section>
      )}
    </div>
  );
}
