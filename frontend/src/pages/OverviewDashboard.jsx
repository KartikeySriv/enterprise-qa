import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Paper,
  Divider,
  CircularProgress,
  Alert,
} from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { apiService } from '../api';

function MetricCard({ title, value, subtitle }) {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" color="textSecondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4" component="div">
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="textSecondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

function OverviewDashboard() {
  const [stats, setStats] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [statsResponse, trainingResponse] = await Promise.all([
          apiService.getStats(),
          apiService.getTrainingHistory(),
        ]);
        
        if (statsResponse.data.error) {
          setError(statsResponse.data.error);
        } else {
          setStats(statsResponse.data);
        }
        
        if (trainingResponse.data.error) {
          setError(trainingResponse.data.error);
        } else {
          setTrainingHistory(trainingResponse.data);
        }
      } catch (err) {
        setError(`Failed to load data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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

  if (!stats) {
    return <Alert severity="warning">No data available</Alert>;
  }

  // Prepare chart data
  const qaComparisonData = [
    {
      metric: 'Exact Match',
      Baseline: (stats.qa_results.baseline.exact_match * 100).toFixed(2),
      'Fine-tuned': (stats.qa_results.fine_tuned.exact_match * 100).toFixed(2),
    },
    {
      metric: 'Token F1',
      Baseline: (stats.qa_results.baseline.token_f1 * 100).toFixed(2),
      'Fine-tuned': (stats.qa_results.fine_tuned.token_f1 * 100).toFixed(2),
    },
  ];

  const trainingData = trainingHistory?.epochs || [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Enterprise QA Research Dashboard
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Domain-Specific Extractive QA Fine-Tuning & Semantic Retrieval
      </Typography>
      <Alert severity="info" sx={{ mb: 3 }}>
        Small synthetic proof-of-concept
      </Alert>

      {/* Data Statistics */}
      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
        Data Statistics
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard title="Source Documents" value={stats.data.source_documents} />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard title="Structured Records" value={stats.data.structured_records} />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard title="QA Examples" value={stats.data.qa_examples} />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard title="Train" value={stats.data.train} />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard title="Validation" value={stats.data.validation} />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard title="Test" value={stats.data.test} />
        </Grid>
      </Grid>

      {/* QA Model Information */}
      <Typography variant="h5" gutterBottom>
        QA Model Information
      </Typography>
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Base Model:</strong> {stats.qa_model.base_model}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Fine-tuned Model:</strong> {stats.qa_model.fine_tuned_model}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Total Parameters:</strong> {stats.qa_model.total_parameters.toLocaleString()}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Epochs:</strong> {stats.qa_model.epochs}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Learning Rate:</strong> {stats.qa_model.learning_rate}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Batch Size:</strong> {stats.qa_model.batch_size}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Device:</strong> {stats.qa_model.device}</Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* QA Results */}
      <Typography variant="h5" gutterBottom>
        QA Results
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Baseline Exact Match" 
            value={`${(stats.qa_results.baseline.exact_match * 100).toFixed(2)}%`} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Fine-tuned Exact Match" 
            value={`${(stats.qa_results.fine_tuned.exact_match * 100).toFixed(2)}%`} 
            subtitle={`+${stats.qa_results.improvement.em_pp} pp`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Baseline Token F1" 
            value={`${(stats.qa_results.baseline.token_f1 * 100).toFixed(2)}%`} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Fine-tuned Token F1" 
            value={`${(stats.qa_results.fine_tuned.token_f1 * 100).toFixed(2)}%`} 
            subtitle={`+${stats.qa_results.improvement.f1_pp} pp`}
          />
        </Grid>
      </Grid>

      {/* QA Results Chart */}
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          QA Model Performance Comparison
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={qaComparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="Baseline" fill="#8884d8" />
            <Bar dataKey="Fine-tuned" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
        <Typography variant="caption" display="block" sx={{ mt: 2 }}>
          Held-out synthetic test set, n=22
        </Typography>
      </Paper>

      {/* Training Loss Chart */}
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Training Loss by Epoch
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#8884d8" name="Training Loss" />
            <Line type="monotone" dataKey="eval_loss" stroke="#82ca9d" name="Validation Loss" />
          </LineChart>
        </ResponsiveContainer>
        <Typography variant="body2" sx={{ mt: 2 }}>
          Training loss continued to decrease while validation loss improved through epoch 2 and increased at epoch 3, 
          indicating a mild overfitting signal.
        </Typography>
      </Paper>

      {/* Retrieval Metrics */}
      <Typography variant="h5" gutterBottom>
        Retrieval Metrics
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Recall@1" 
            value={`${(stats.retrieval.recall_at_1 * 100).toFixed(2)}%`} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Recall@3" 
            value={`${(stats.retrieval.recall_at_3 * 100).toFixed(2)}%`} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Recall@5" 
            value={`${(stats.retrieval.recall_at_5 * 100).toFixed(2)}%`} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Embedding Dimension" 
            value={stats.retrieval.dimension} 
          />
        </Grid>
      </Grid>

      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Retrieval System Details
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Embedding Model:</strong> {stats.retrieval.embedding_model}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body1"><strong>Records Embedded:</strong> {stats.retrieval.records_embedded}</Typography>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}

export default OverviewDashboard;