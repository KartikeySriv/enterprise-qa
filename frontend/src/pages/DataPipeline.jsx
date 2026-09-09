import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';

function DataPipeline() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Pipeline & Technical Details
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Overview of the data processing pipeline and technical implementation
      </Typography>

      {/* QA Pipeline */}
      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
        QA Data Pipeline
      </Typography>
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Box display="flex" flexDirection="column" alignItems="center">
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                10 AsciiDoc Documents
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/raw/*.adoc
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                AsciiDoc Parser
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/process_documents.py
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                154 Structured Records
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/processed/sections.jsonl
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                150 QA Examples
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/qa_dataset/processed/all_examples.jsonl
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                105 Train / 23 Validation / 22 Test
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                Document-stratified split
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Tokenization
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                qa/tokenize_qa.py
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                PyTorch Tensors
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/qa_dataset/processed/tokenized/
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                tinyRoBERTa Fine-tuning
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                qa/train.py → models/tinyroberta-aurelia-qa/
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Paper>

      {/* Retrieval Pipeline */}
      <Typography variant="h5" gutterBottom>
        Retrieval Pipeline
      </Typography>
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Box display="flex" flexDirection="column" alignItems="center">
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                154 Structured Records
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/processed/sections.jsonl
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                BGE-small-en-v1.5
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                retrieval/embed.py
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                154 × 384 Embeddings
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                data/processed/embeddings.npy
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Cosine Similarity
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                L2-normalized vectors
              </Typography>
            </CardContent>
          </Card>
          
          <ArrowDownwardIcon sx={{ my: 1 }} />
          
          <Card elevation={1} sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
            <CardContent>
              <Typography variant="h6" align="center">
                Top-K Semantic Retrieval
              </Typography>
              <Typography variant="body2" color="textSecondary" align="center">
                retrieval/search.py
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Paper>

      {/* Technical Details */}
      <Typography variant="h5" gutterBottom>
        Technical Details
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Frameworks
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText primary="Python" />
              </ListItem>
              <ListItem>
                <ListItemText primary="PyTorch" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Hugging Face Transformers" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Hugging Face Datasets" />
              </ListItem>
              <ListItem>
                <ListItemText primary="Sentence Transformers" />
              </ListItem>
              <ListItem>
                <ListItemText primary="NumPy" />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Models
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="deepset/tinyroberta-squad2" 
                  secondary="Base QA model"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="BAAI/bge-small-en-v1.5" 
                  secondary="Embedding model (384-dim)"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Important Files
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Document Processing
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="data/process_documents.py" 
                      secondary="AsciiDoc parser + JSONL writer"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="data/processed/sections.jsonl" 
                      secondary="154 structured section records"
                    />
                  </ListItem>
                </List>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  QA Processing
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="data/qa_dataset/processed/all_examples.jsonl" 
                      secondary="150 QA examples"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="data/qa_dataset/processed/train.jsonl" 
                      secondary="105 training examples"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="data/qa_dataset/processed/validation.jsonl" 
                      secondary="23 validation examples"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="data/qa_dataset/processed/test.jsonl" 
                      secondary="22 test examples"
                    />
                  </ListItem>
                </List>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  QA Training
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="qa/tokenize_qa.py" 
                      secondary="Tokenization + span alignment"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="qa/train.py" 
                      secondary="Fine-tuning script"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="qa/evaluate_qa.py" 
                      secondary="Baseline vs fine-tuned evaluation"
                    />
                  </ListItem>
                </List>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Retrieval
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="retrieval/embed.py" 
                      secondary="BGE embedding generation"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="retrieval/search.py" 
                      secondary="Interactive semantic search"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="retrieval/evaluate_retrieval.py" 
                      secondary="Recall@K evaluation"
                    />
                  </ListItem>
                </List>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Data Files
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="data/processed/embeddings.npy" 
                      secondary="154 × 384 embedding matrix"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="data/processed/embedding_metadata.json" 
                      secondary="Vector → record mapping"
                    />
                  </ListItem>
                </List>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Evaluation Results
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="models/tinyroberta-aurelia-qa/" 
                      secondary="Fine-tuned model checkpoint"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="evaluation/qa_evaluation.json" 
                      secondary="QA performance comparison"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="evaluation/retrieval_evaluation.json" 
                      secondary="Retrieval Recall@K metrics"
                    />
                  </ListItem>
                </List>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default DataPipeline;