import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Alert,
  Divider,
} from '@mui/material';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';

function ExperimentalInterpretation() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Experimental Interpretation & Future Work
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Research findings and next steps for the enterprise QA system
      </Typography>

      {/* QA Experiment Summary */}
      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
        QA Experiment Results
      </Typography>
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="body1" paragraph>
          Fine-tuning the pretrained extractive QA model on 105 domain-specific examples improved 
          Exact Match from 50.00% to 86.36% and Token F1 from 78.91% to 95.37% on the same 22 
          held-out test questions.
        </Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            <strong>Key Insight:</strong> Domain-specific fine-tuning significantly improves the model's 
            ability to extract accurate answers from banking policy documents, particularly for 
            regulatory figures, defined terms, and named entities.
          </Typography>
        </Alert>
      </Paper>

      {/* Retrieval Experiment Summary */}
      <Typography variant="h5" gutterBottom>
        Retrieval Experiment Results
      </Typography>
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="body1" paragraph>
          BGE-small semantic retrieval achieved Recall@1 = 63.64%, Recall@3 = 86.36%, and 
          Recall@5 = 90.91% on the same controlled test set.
        </Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            <strong>Key Insight:</strong> The semantic retriever successfully surfaces the correct 
            source section within the top-5 results for over 90% of test questions, making it a 
            viable first-stage retriever for a full RAG pipeline.
          </Typography>
        </Alert>
      </Paper>

      {/* Important Distinction */}
      <Paper elevation={3} sx={{ p: 3, mb: 4, bgcolor: 'warning.light' }}>
        <Typography variant="h6" gutterBottom color="warning.dark">
          Important Experimental Distinction
        </Typography>
        <Typography variant="body1" paragraph>
          <strong>Fine-tuning affects the QA model's answer extraction behavior.</strong>
        </Typography>
        <Typography variant="body1" paragraph>
          <strong>Semantic retrieval is evaluated independently.</strong>
        </Typography>
        <Typography variant="body2" color="textSecondary">
          The fine-tuning experiment changed how the QA model extracts answer spans from provided 
          context. The retrieval experiment evaluated how well BGE embeddings find relevant documents. 
          These are separate components that can be combined, but the current results do not represent 
          an end-to-end RAG system.
        </Typography>
      </Paper>

      {/* Future End-to-End Pipeline */}
      <Typography variant="h5" gutterBottom>
        Experimental Next Step: End-to-End RAG Pipeline
      </Typography>
      <Alert severity="warning" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>NOT YET EVALUATED:</strong> The following pipeline represents the next experimental 
          step and has not been benchmarked.
        </Typography>
      </Alert>

      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Box display="flex" flexDirection="column" alignItems="center">
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                User Question
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Natural language query
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                BGE Semantic Retrieval
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Find top-K relevant records
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Top-K Relevant Records
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Retrieved document sections
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Construct QA Context
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Concatenate retrieved passages
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Fine-tuned tinyRoBERTa
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Extract answer span from context
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Answer Span
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Final extracted answer
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Divider sx={{ my: 3 }} />
        
        <Alert severity="info">
          <Typography variant="body2">
            <strong>Next experiment:</strong> Evaluate the integrated retrieval → fine-tuned QA pipeline 
            to measure end-to-end performance on the held-out test set.
          </Typography>
        </Alert>
      </Paper>

      {/* Research Questions */}
      <Typography variant="h5" gutterBottom>
        Open Research Questions
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Retrieval Quality
            </Typography>
            <Typography variant="body2" paragraph>
              How does retrieval performance impact end-to-end QA accuracy? What is the optimal 
              top-K value for context construction?
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Context Length
            </Typography>
            <Typography variant="body2" paragraph>
              How does the amount of retrieved context affect the fine-tuned QA model's performance? 
              Is there an optimal context window size?
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Error Analysis
            </Typography>
            <Typography variant="body2" paragraph>
              What types of questions does the integrated system still fail on? Are these failures 
              due to retrieval or QA extraction limitations?
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Scalability
            </Typography>
            <Typography variant="body2" paragraph>
              How does the system perform with larger corpora? What scaling strategies are needed 
              for production deployment?
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ExperimentalInterpretation;