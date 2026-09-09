import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SearchIcon from '@mui/icons-material/Search';
import { apiService } from '../api';

function SemanticSearch() {
  const [query, setQuery] = useState('What is the maximum Loan-to-Value Ratio for self-employed applicants?');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    try {
      setLoading(true);
      setError(null);
      const response = await apiService.searchRetrieval(query, 5);
      
      if (response.data.error) {
        setError(response.data.error);
      } else {
        setResults(response.data);
      }
    } catch (err) {
      setError(`Search failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Semantic Search
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Search the 154 structured banking policy records using BGE-small-en-v1.5 embeddings
      </Typography>

      {/* Search Input */}
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <TextField
          fullWidth
          label="Enter your query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          multiline
          rows={2}
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          startIcon={<SearchIcon />}
          onClick={handleSearch}
          disabled={loading}
          fullWidth
        >
          {loading ? 'Searching...' : 'Search'}
        </Button>
      </Paper>

      {loading && (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {results && results.results.length === 0 && (
        <Alert severity="info" sx={{ mb: 4 }}>
          No sufficiently similar source records found (similarity less than 0.5). 
          {results.total_candidates > 0 && ` Searched ${results.total_candidates} candidates.`}
        </Alert>
      )}

      {results && results.results.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Search Results
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Retrieval method: BGE-small-en-v1.5 + cosine similarity
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Found {results.filtered_results} relevant results (similarity ≥ 0.5) out of {results.total_candidates} candidates
          </Typography>

          {results.results.map((result, index) => (
            <Card key={result.record_id} elevation={2} sx={{ mb: 3 }}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Chip 
                    label={`#${result.rank}`} 
                    color="primary" 
                    size="small" 
                    sx={{ mr: 2, fontWeight: 'bold' }}
                  />
                  <Typography variant="h6" component="div">
                    Similarity: {result.similarity.toFixed(6)}
                  </Typography>
                </Box>

                <LinearProgress 
                  variant="determinate" 
                  value={result.similarity * 100} 
                  sx={{ mb: 2 }}
                />

                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      <strong>Record ID:</strong> {result.record_id}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      <strong>Document:</strong> {result.document}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      <strong>Section:</strong> {result.section}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">
                      <strong>Subsection:</strong> {result.subsection || 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="body2" color="textSecondary">
                      <strong>Source:</strong> {result.source}
                    </Typography>
                  </Grid>
                </Grid>

                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="body1">View Retrieved Text</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                      {result.text}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              </CardContent>
            </Card>
          ))}
        </Box>
      )}
    </Box>
  );
}

export default SemanticSearch;