import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';

// Import pages
import OverviewDashboard from './pages/OverviewDashboard';
import SemanticSearch from './pages/SemanticSearch';
import QAComparison from './pages/QAComparison';
import DataPipeline from './pages/DataPipeline';
import ExperimentalInterpretation from './pages/ExperimentalInterpretation';
import ResearchLimitations from './pages/ResearchLimitations';

// Create a professional theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '0.875rem',
      fontWeight: 600,
    },
  },
});

function Navigation() {
  const navItems = [
    { label: 'Overview', path: '/' },
    { label: 'Semantic Search', path: '/semantic-search' },
    { label: 'QA Comparison', path: '/qa-comparison' },
    { label: 'Data Pipeline', path: '/data-pipeline' },
    { label: 'Interpretation', path: '/interpretation' },
    { label: 'Limitations', path: '/limitations' },
  ];

  return (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Enterprise QA Research Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {navItems.map((item) => (
            <Typography
              key={item.path}
              component="a"
              href={item.path}
              sx={{
                color: 'white',
                textDecoration: 'none',
                fontSize: '0.875rem',
                '&:hover': {
                  textDecoration: 'underline',
                },
              }}
            >
              {item.label}
            </Typography>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Navigation />
          <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flex: 1 }}>
            <Routes>
              <Route path="/" element={<OverviewDashboard />} />
              <Route path="/semantic-search" element={<SemanticSearch />} />
              <Route path="/qa-comparison" element={<QAComparison />} />
              <Route path="/data-pipeline" element={<DataPipeline />} />
              <Route path="/interpretation" element={<ExperimentalInterpretation />} />
              <Route path="/limitations" element={<ResearchLimitations />} />
            </Routes>
          </Container>
          <Box
            component="footer"
            sx={{
              py: 3,
              px: 2,
              mt: 'auto',
              backgroundColor: 'grey.100',
            }}
          >
            <Container maxWidth="lg">
              <Typography variant="body2" color="text.secondary" align="center">
                Enterprise QA Research Dashboard - Small synthetic proof-of-concept
              </Typography>
            </Container>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;