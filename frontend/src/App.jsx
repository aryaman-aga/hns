import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Landing from './pages/Landing';
import Auth from './pages/Auth';
import PractitionerDashboard from './pages/PractitionerDashboard';
import PatientDashboard from './pages/PatientDashboard';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Navbar />
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/auth" element={<Auth />} />
            <Route path="/practitioner" element={<PractitionerDashboard />} />
            <Route path="/patient" element={<PatientDashboard />} />
          </Routes>
        </main>
        <Footer />
      </Router>
    </AuthProvider>
  );
}

export default App;
