import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Scan, Heart } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import ScanUploader from '../components/ScanUploader';
import ResultsPanel from '../components/ResultsPanel';
import './PractitionerDashboard.css'; /* Reuse same dashboard styles */

const API_URL = 'http://localhost:5000';

export default function PatientDashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [task, setTask] = useState('pneumonia');
  const [xaiMethod, setXaiMethod] = useState('gradcam');
  const [imageFile, setImageFile] = useState(null);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [aiExplanation, setAiExplanation] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  if (!user) {
    navigate('/auth');
    return null;
  }

  const handleAnalyze = async () => {
    if (!imageFile) return;
    setAnalyzing(true);
    setResult(null);
    setAiExplanation('');

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('task', task);
      formData.append('method', 'gradcam');

      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Analysis failed');
      const data = await res.json();
      setResult(data);

      // Auto-generate patient-friendly explanation
      setAiLoading(true);
      try {
        const explainRes = await fetch(`${API_URL}/api/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task,
            label: data.label,
            confidence: data.confidence,
            class_probs: data.class_probs,
            role: 'patient',
          }),
        });
        if (explainRes.ok) {
          const explainData = await explainRes.json();
          setAiExplanation(explainData.explanation);
        }
      } catch {}
      setAiLoading(false);
    } catch (err) {
      console.error(err);
      alert('Analysis failed. Make sure the Flask API (api.py) is running.');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleGenerateExplanation = async () => {
    if (!result) return;
    setAiLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          label: result.label,
          confidence: result.confidence,
          class_probs: result.class_probs,
          role: 'patient',
        }),
      });

      if (!res.ok) throw new Error('Explanation generation failed');
      const data = await res.json();
      setAiExplanation(data.explanation);
    } catch (err) {
      console.error(err);
      setAiExplanation('⚠️ Could not generate AI explanation. Make sure Ollama is running (ollama run llama3).');
    } finally {
      setAiLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="bg-decoration">
        <div className="bg-blob bg-blob-1" />
        <div className="bg-blob bg-blob-2" />
      </div>

      <motion.div
        className="dashboard-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>🏥 Patient Dashboard</h1>
        <p>Upload your medical scan for AI-guided health assessment and doctor consultation advice</p>
      </motion.div>

      <div className="dashboard-content">
        {/* Sidebar */}
        <motion.div
          className="dashboard-sidebar"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="dashboard-sidebar-card">
            <h3>📋 Upload Your Scan</h3>
            <ScanUploader
              onImageSelect={setImageFile}
              task={task}
              setTask={setTask}
              xaiMethod={xaiMethod}
              setXaiMethod={setXaiMethod}
              showXai={false}
            />

            <button
              className="btn btn-primary btn-large analyze-btn"
              onClick={handleAnalyze}
              disabled={!imageFile || analyzing}
            >
              {analyzing ? (
                'Checking...'
              ) : (
                <>
                  <Heart size={18} /> Check My Scan
                </>
              )}
            </button>
          </div>

          {/* Patient Info */}
          <div className="patient-info-card" style={{ marginTop: '16px' }}>
            <h4>Your Profile</h4>
            <div className="patient-info-row">
              <span>Name</span>
              <span>{user.name}</span>
            </div>
            <div className="patient-info-row">
              <span>Account Type</span>
              <span>Patient</span>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="patient-info-card" style={{ marginTop: '12px', background: 'rgba(255, 152, 0, 0.06)', border: '1px solid rgba(255, 152, 0, 0.2)' }}>
            <h4 style={{ color: 'var(--warning)' }}>⚠️ Disclaimer</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              This AI tool is for informational purposes only. It does not replace 
              professional medical advice, diagnosis, or treatment. Always consult 
              a qualified healthcare provider.
            </p>
          </div>
        </motion.div>

        {/* Main */}
        <motion.div
          className="dashboard-main"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="dashboard-main-card">
            {analyzing ? (
              <div className="analysis-loading">
                <div className="analysis-spinner" />
                <h3>Checking Your Scan...</h3>
                <p>Our AI is evaluating your medical image. This may take a few seconds.</p>
              </div>
            ) : result ? (
              <ResultsPanel
                result={result}
                mode="patient"
                onGenerateExplanation={handleGenerateExplanation}
                aiExplanation={aiExplanation}
                aiLoading={aiLoading}
              />
            ) : (
              <div className="dashboard-empty">
                <div className="dashboard-empty-icon">💆</div>
                <h3>Upload Your Scan</h3>
                <p>Choose a disease model, upload your scan, and our AI will guide you on whether you need to see a doctor</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
