import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Scan, Brain } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import ScanUploader from '../components/ScanUploader';
import ResultsPanel from '../components/ResultsPanel';
import './PractitionerDashboard.css';

const API_URL = 'http://localhost:5000';

export default function PractitionerDashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [task, setTask] = useState('pneumonia');
  const [xaiMethod, setXaiMethod] = useState('both');
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
      formData.append('method', xaiMethod);

      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Analysis failed');
      const data = await res.json();
      setResult(data);
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
          role: 'practitioner',
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
        <h1>👨‍⚕️ Practitioner Dashboard</h1>
        <p>Upload medical scans for AI-powered severity analysis with dual-granularity explainability</p>
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
            <h3>🔬 Scan Configuration</h3>
            <ScanUploader
              onImageSelect={setImageFile}
              task={task}
              setTask={setTask}
              xaiMethod={xaiMethod}
              setXaiMethod={setXaiMethod}
              showXai={true}
            />

            <button
              className="btn btn-primary btn-large analyze-btn"
              onClick={handleAnalyze}
              disabled={!imageFile || analyzing}
            >
              {analyzing ? (
                'Analyzing...'
              ) : (
                <>
                  <Scan size={18} /> Analyze Scan
                </>
              )}
            </button>
          </div>

          {/* Practitioner Info */}
          <div className="patient-info-card" style={{ marginTop: '16px' }}>
            <h4>Practitioner Profile</h4>
            <div className="patient-info-row">
              <span>Name</span>
              <span>{user.name}</span>
            </div>
            <div className="patient-info-row">
              <span>Role</span>
              <span>Medical Practitioner</span>
            </div>
            {user.specialization && (
              <div className="patient-info-row">
                <span>Specialization</span>
                <span>{user.specialization}</span>
              </div>
            )}
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
                <h3>Analyzing Medical Scan...</h3>
                <p>Running SE-ResNet-18 inference with {xaiMethod === 'both' ? 'Dual-Granularity' : xaiMethod.toUpperCase()} explainability</p>
              </div>
            ) : result ? (
              <ResultsPanel
                result={result}
                mode="practitioner"
                onGenerateExplanation={handleGenerateExplanation}
                aiExplanation={aiExplanation}
                aiLoading={aiLoading}
              />
            ) : (
              <div className="dashboard-empty">
                <div className="dashboard-empty-icon">🩺</div>
                <h3>No Scan Analyzed Yet</h3>
                <p>Upload a medical image and click "Analyze Scan" to begin AI-powered diagnosis</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
