import React from 'react';
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  Alert,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';

function ResearchLimitations() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Research Limitations
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Important constraints and considerations for interpreting the experimental results
      </Typography>

      <Alert severity="warning" icon={<WarningIcon />} sx={{ mb: 4 }}>
        <Typography variant="body2">
          This is a research prototype with synthetic data. Results should not be generalized to 
          production systems or real-world banking applications without further validation.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="warning.main">
                Data Limitations
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Small Corpus Size"
                    secondary="Only 10 synthetic banking policy documents (154 structured records)"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Limited QA Examples"
                    secondary="150 total QA examples with only 22 held-out test questions"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Synthetic Data"
                    secondary="All data is fictional and synthetic, not real bank policies or customer data"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Single Domain"
                    secondary="Experiments limited to banking domain; generalizability untested"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="warning.main">
                Evaluation Limitations
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Separate Component Evaluation"
                    secondary="Retrieval and QA are evaluated independently, not as an integrated system"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="No End-to-End RAG"
                    secondary="The full retrieval → QA pipeline has not been benchmarked"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Static Test Set"
                    secondary="Results based on a single frozen test set; no cross-validation"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Limited Metrics"
                    secondary="Only Exact Match and Token F1 for QA; Recall@K for retrieval"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="warning.main">
                Model Limitations
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Small Model"
                    secondary="tinyRoBERTa has limited capacity compared to larger modern models"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Extractive Only"
                    secondary="Model can only extract answer spans, not generate abstractive answers"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Context Window"
                    secondary="512 token limit may truncate long documents"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="No Fine-tuning of Retrieval"
                    secondary="BGE model used as-is; no domain-specific adaptation"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="warning.main">
                Implementation Limitations
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="CPU Training"
                    secondary="Fine-tuning performed on CPU; GPU training not tested"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Local Deployment"
                    secondary="No cloud deployment or production optimization"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="No Persistence"
                    secondary="No database or vector store; all data in memory/files"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Single Language"
                    secondary="Only English language support tested"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3, bgcolor: 'info.light' }}>
            <Typography variant="h6" gutterBottom color="info.dark">
              Appropriate Use Cases
            </Typography>
            <Typography variant="body1" paragraph>
              This research prototype is suitable for:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText primary="Academic research and educational demonstrations" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Understanding RAG system components and interactions" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Baseline experiments for further research" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Prototyping and proof-of-concept development" />
              </ListItem>
            </List>
            <Divider sx={{ my: 2 }} />
            <Typography variant="body1" paragraph>
              This research prototype is NOT suitable for:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText primary="Production banking systems or customer-facing applications" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Making real financial or regulatory decisions" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Handling sensitive customer data or personally identifiable information" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Compliance-critical applications without extensive validation" />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      <Alert severity="info" sx={{ mt: 4 }}>
        <Typography variant="body2">
          <strong>Future Work:</strong> Address these limitations through larger datasets, diverse domains, 
          end-to-end evaluation, more comprehensive metrics, and production-grade infrastructure.
        </Typography>
      </Alert>
    </Box>
  );
}

export default ResearchLimitations;