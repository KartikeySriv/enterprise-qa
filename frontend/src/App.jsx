import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import InsightsRoundedIcon from '@mui/icons-material/InsightsRounded';
import DashboardRoundedIcon from '@mui/icons-material/DashboardRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import CompareArrowsRoundedIcon from '@mui/icons-material/CompareArrowsRounded';
import AccountTreeRoundedIcon from '@mui/icons-material/AccountTreeRounded';
import ScienceRoundedIcon from '@mui/icons-material/ScienceRounded';
import WarningAmberRoundedIcon from '@mui/icons-material/WarningAmberRounded';

// Import pages
import OverviewDashboard from './pages/OverviewDashboard';
import SemanticSearch from './pages/SemanticSearch';
import QAComparison from './pages/QAComparison';
import DataPipeline from './pages/DataPipeline';
import ExperimentalInterpretation from './pages/ExperimentalInterpretation';
import ResearchLimitations from './pages/ResearchLimitations';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#635bff',
      light: '#867fff',
      dark: '#443bd1',
    },
    secondary: {
      main: '#ec4899',
    },
    background: {
      default: '#f7f8fc',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Plus Jakarta Sans", "Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    button: {
      fontWeight: 700,
      textTransform: 'none',
    },
    h1: {
      fontSize: '2.7rem',
      fontWeight: 800,
      letterSpacing: '-0.04em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 800,
      letterSpacing: '-0.03em',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 800,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 800,
    },
    h5: {
      fontSize: '1rem',
      fontWeight: 800,
    },
    h6: {
      fontSize: '0.875rem',
      fontWeight: 800,
      letterSpacing: '-0.01em',
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          border: '1px solid rgba(99, 91, 255, 0.08)',
          boxShadow: '0 12px 30px rgba(42, 47, 77, 0.07)',
          transition: 'transform 180ms ease, box-shadow 180ms ease',
          '&:hover': {
            transform: 'translateY(-3px)',
            boxShadow: '0 18px 38px rgba(42, 47, 77, 0.12)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          border: '1px solid rgba(99, 91, 255, 0.08)',
          boxShadow: '0 12px 30px rgba(42, 47, 77, 0.06)',
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '11px 18px',
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: '#fbfbfe',
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderWidth: 2,
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 14,
        },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: '#f4f3ff',
        },
      },
    },
  },
});

function Navigation() {
  const navItems = [
    { label: 'Overview', path: '/', icon: <DashboardRoundedIcon /> },
    { label: 'Semantic Search', path: '/semantic-search', icon: <SearchRoundedIcon /> },
    { label: 'QA Comparison', path: '/qa-comparison', icon: <CompareArrowsRoundedIcon /> },
    { label: 'Data Pipeline', path: '/data-pipeline', icon: <AccountTreeRoundedIcon /> },
    { label: 'Interpretation', path: '/interpretation', icon: <ScienceRoundedIcon /> },
    // { label: 'Limitations', path: '/limitations', icon: <WarningAmberRoundedIcon /> },
  ];

  return (
    <AppBar position="sticky" elevation={0}>
      <Toolbar sx={{ maxWidth: 1400, width: '100%', mx: 'auto', px: { xs: 2, md: 4 }, py: 1.25 }}>
        <Stack direction="row" alignItems="center" spacing={1.25} sx={{ mr: { xs: 2, md: 5 }, flexShrink: 0 }}>
          <Box className="brand-mark">
            <InsightsRoundedIcon />
          </Box>
          <Box>
            <Typography variant="h6" component="div" sx={{ lineHeight: 1.1, color: 'white' }}>
              Enterprise QA
            </Typography>
            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,.65)', letterSpacing: '.08em' }}>
              RESEARCH STUDIO
            </Typography>
          </Box>
        </Stack>
        <Box className="nav-scroll" sx={{ display: 'flex', gap: 0.5, overflowX: 'auto' }}>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {item.icon}
              {item.label}
            </NavLink>
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
          <Box className="page-glow" />
          <Container maxWidth="xl" sx={{ mt: { xs: 3, md: 5 }, mb: 6, flex: 1, position: 'relative' }}>
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
              py: 3.5,
              px: 2,
              mt: 'auto',
              backgroundColor: '#171936',
              color: 'rgba(255,255,255,.72)',
            }}
          >
            <Container maxWidth="lg">
              <Divider sx={{ mb: 2.5, borderColor: 'rgba(255,255,255,.12)' }} />
              <Typography variant="body2" align="center">
                Enterprise QA Research Dashboard <span className="footer-dot">•</span> Small synthetic proof-of-concept
              </Typography>
            </Container>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;