import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
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
  
  // Semantic search
  searchRetrieval: (query, topK = 5) => 
    api.post('/api/retrieval/search', { query, top_k: topK }),
};

export default apiService;