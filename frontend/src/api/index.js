import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minute timeout for model loading
});

export const apiService = {
  // Health check
  health: () => api.get('/api/health'),
  
  // Statistics
  getStats: () => api.get('/api/stats'),
  
  // QA comparison
  getQAComparison: () => api.get('/api/qa/comparison'),
  
  // Retrieval metrics
  getRetrievalMetrics: () => api.get('/api/retrieval/metrics'),
  
  // Training history
  getTrainingHistory: () => api.get('/api/training/history'),
  
  // Semantic search with extended timeout
  searchRetrieval: (query, topK = 5) => 
    api.post('/api/retrieval/search', { query, top_k: topK }, { timeout: 180000 }), // 3 minute timeout
};

export default apiService;