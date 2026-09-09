import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  TextField,
  CircularProgress,
  Alert,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { apiService } from '../api';

function getStatus(baselineEM, fineTunedEM) {
  if (baselineEM === 0 && fineTunedEM === 1) {
    return { label: 'Improved', color: 'success' };
  } else if (baselineEM === 1 && fineTunedEM === 1) {
    return { label: 'Correct → Correct', color: 'info' };
  } else if (baselineEM === 0 && fineTunedEM === 0) {
    return { label: 'Still Incorrect', color: 'warning' };
  } else if (baselineEM === 1 && fineTunedEM === 0) {
    return { label: 'Regression', color: 'error' };
  }
  return { label: 'Unknown', color: 'default' };
}

function QAComparison() {
  const [qaData, setQaData] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await apiService.getQAComparison();
        
        if (response.data.error) {
          setError(response.data.error);
        } else {
          setQaData(response.data);
          setFilteredData(response.data);
        }
      } catch (err) {
        setError(`Failed to load QA comparison data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    if (!qaData) return;

    let filtered = [...qaData.fine_tuned.predictions];

    // Apply status filter
    if (filter !== 'all') {
      filtered = filtered.filter((item) => {
        const baselineItem = qaData.baseline.predictions.find(
          (b) => b.id === item.id
        );
        if (!baselineItem) return false;

        const status = getStatus(baselineItem.exact_match, item.exact_match);
        
        switch (filter) {
          case 'improved':
            return status.label === 'Improved';
          case 'correct_correct':
            return status.label === 'Correct → Correct';
          case 'still_incorrect':
            return status.label === 'Still Incorrect';
          case 'regression':
            return status.label === 'Regression';
          default:
            return true;
        }
      });
    }

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter((item) =>
        item.question.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredData({ ...qaData, fine_tuned: { ...qaData.fine_tuned, predictions: filtered } });
  }, [filter, searchQuery, qaData]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!qaData) {
    return <Alert severity="warning">No data available</Alert>;
  }

  // Calculate summary statistics
  const totalQuestions = qaData.fine_tuned.predictions.length;
  let improvedCount = 0;
  let regressionCount = 0;
  let baselineEMCount = 0;
  let fineTunedEMCount = 0;

  qaData.fine_tuned.predictions.forEach((fineTunedItem) => {
    const baselineItem = qaData.baseline.predictions.find((b) => b.id === fineTunedItem.id);
    if (baselineItem) {
      const status = getStatus(baselineItem.exact_match, fineTunedItem.exact_match);
      if (status.label === 'Improved') improvedCount++;
      if (status.label === 'Regression') regressionCount++;
      if (baselineItem.exact_match === 1) baselineEMCount++;
      if (fineTunedItem.exact_match === 1) fineTunedEMCount++;
    }
  });

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        QA Before vs After Fine-tuning
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Comparison of baseline and fine-tuned models on the frozen 22-question test set
      </Typography>

      {/* Summary Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4">{totalQuestions}</Typography>
            <Typography variant="body2" color="textSecondary">
              Total Questions
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="success.main">{improvedCount}</Typography>
            <Typography variant="body2" color="textSecondary">
              Improved
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="error.main">{regressionCount}</Typography>
            <Typography variant="body2" color="textSecondary">
              Regressions
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4">{baselineEMCount}</Typography>
            <Typography variant="body2" color="textSecondary">
              Baseline Exact Matches
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4">{fineTunedEMCount}</Typography>
            <Typography variant="body2" color="textSecondary">
              Fine-tuned Exact Matches
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Filters */}
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth>
              <InputLabel>Filter by Status</InputLabel>
              <Select
                value={filter}
                label="Filter by Status"
                onChange={(e) => setFilter(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="improved">Improved</MenuItem>
                <MenuItem value="correct_correct">Correct → Correct</MenuItem>
                <MenuItem value="still_incorrect">Still Incorrect</MenuItem>
                <MenuItem value="regression">Regression</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={8}>
            <TextField
              fullWidth
              label="Search questions"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by question text..."
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Results Table */}
      <Paper elevation={2}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Question</TableCell>
                <TableCell>Reference Answer</TableCell>
                <TableCell>Baseline Prediction</TableCell>
                <TableCell>Fine-tuned Prediction</TableCell>
                <TableCell>Baseline EM</TableCell>
                <TableCell>Baseline F1</TableCell>
                <TableCell>Fine-tuned EM</TableCell>
                <TableCell>Fine-tuned F1</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredData.fine_tuned.predictions.map((fineTunedItem) => {
                const baselineItem = qaData.baseline.predictions.find(
                  (b) => b.id === fineTunedItem.id
                );
                const status = baselineItem 
                  ? getStatus(baselineItem.exact_match, fineTunedItem.exact_match)
                  : { label: 'Unknown', color: 'default' };

                return (
                  <TableRow key={fineTunedItem.id}>
                    <TableCell sx={{ maxWidth: 300 }}>
                      <Typography variant="body2" noWrap>
                        {fineTunedItem.question}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 200 }}>
                      <Typography variant="body2" noWrap>
                        {fineTunedItem.reference_answer}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 200 }}>
                      <Typography variant="body2" noWrap>
                        {baselineItem?.prediction || 'N/A'}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 200 }}>
                      <Typography variant="body2" noWrap>
                        {fineTunedItem.prediction}
                      </Typography>
                    </TableCell>
                    <TableCell>{baselineItem?.exact_match || 0}</TableCell>
                    <TableCell>{baselineItem?.f1?.toFixed(2) || '0.00'}</TableCell>
                    <TableCell>{fineTunedItem.exact_match}</TableCell>
                    <TableCell>{fineTunedItem.f1.toFixed(2)}</TableCell>
                    <TableCell>
                      <Chip label={status.label} color={status.color} size="small" />
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Typography variant="caption" display="block" sx={{ mt: 2 }}>
        Showing {filteredData.fine_tuned.predictions.length} of {totalQuestions} questions
      </Typography>
    </Box>
  );
}

export default QAComparison;