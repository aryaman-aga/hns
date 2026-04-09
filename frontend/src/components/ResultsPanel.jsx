import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, AlertCircle, Brain, Download, FileText } from 'lucide-react';
import './ResultsPanel.css';

function getSeverity(confidence, label) {
  // For positive classes (Pneumonia, Malignant), high confidence = high severity
  const positiveLabels = ['Pneumonia', 'Malignant'];
  const isPositive = positiveLabels.includes(label);

  if (isPositive && confidence > 0.85) return 'critical';
  if (isPositive && confidence > 0.6) return 'moderate';
  return 'low';
}

function SeverityCard({ severity, label, confidence }) {
  const config = {
    critical: { icon: '🔴', title: 'High Severity', desc: `The model strongly predicts ${label} with ${(confidence * 100).toFixed(1)}% confidence. Immediate clinical review recommended.` },
    moderate: { icon: '🟡', title: 'Moderate Severity', desc: `${label} detected with ${(confidence * 100).toFixed(1)}% confidence. Further evaluation suggested.` },
    low: { icon: '🟢', title: 'Low Severity', desc: `${label} with ${(confidence * 100).toFixed(1)}% confidence. Findings appear within normal range.` },
  };
  const c = config[severity];

  return (
    <motion.div
      className={`severity-card severity-${severity}`}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="severity-icon">{c.icon}</div>
      <div className="severity-text">
        <h3>{c.title}</h3>
        <p>{c.desc}</p>
      </div>
    </motion.div>
  );
}

export default function ResultsPanel({ result, mode = 'practitioner', onGenerateExplanation, aiExplanation, aiLoading }) {
  if (!result) return null;

  const { label, confidence, class_probs, heatmaps } = result;
  const severity = getSeverity(confidence, label);
  const shouldConsult = severity === 'critical' || severity === 'moderate';

  return (
    <div className="results-panel">
      {/* Severity */}
      <SeverityCard severity={severity} label={label} confidence={confidence} />

      {mode === 'patient' ? (
        /* Patient Mode */
        <div className="patient-guidance">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className={`guidance-verdict ${shouldConsult ? 'consult' : 'monitor'}`}>
              {shouldConsult ? '⚠️ Please Consult a Doctor' : '✅ Low Risk — Continue Monitoring'}
            </div>
            <p className="guidance-message">
              {shouldConsult
                ? `Our AI model has detected potential signs of ${label} in your scan. We strongly recommend scheduling a consultation with a healthcare professional for a thorough evaluation.`
                : `Based on the analysis, your scan appears to show normal findings. However, please continue regular check-ups and consult a doctor if you experience any symptoms.`
              }
            </p>
          </motion.div>

          {/* Doctor Brief for Patient */}
          <div className="doctor-brief">
            <h4>📋 Brief for Your Doctor</h4>
            <p>
              <strong>AI Prediction:</strong> {label} ({(confidence * 100).toFixed(1)}% confidence)<br />
              <strong>Severity Assessment:</strong> {severity.charAt(0).toUpperCase() + severity.slice(1)}<br />
              <strong>Model:</strong> SE-ResNet-18 with Dual-Granularity XAI<br />
              <strong>Recommendation:</strong> {shouldConsult ? 'Clinical review recommended' : 'Routine monitoring'}
            </p>
          </div>
        </div>
      ) : (
        /* Practitioner Mode */
        <>
          {/* Prediction */}
          <motion.div
            className="prediction-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="prediction-header">
              <div className="prediction-label">{label}</div>
              <div className={`confidence-badge badge badge-${severity === 'low' ? 'success' : severity === 'moderate' ? 'warning' : 'danger'}`}>
                {(confidence * 100).toFixed(1)}%
              </div>
            </div>

            {class_probs && Object.entries(class_probs).map(([cls, prob]) => (
              <div key={cls} className="class-prob">
                <div className="class-prob-header">
                  <span className="class-prob-name">{cls}</span>
                  <span className="class-prob-value">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar">
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${prob * 100}%` }}
                    transition={{ duration: 1, delay: 0.3 }}
                  />
                </div>
              </div>
            ))}
          </motion.div>
        </>
      )}

      {/* Heatmaps */}
      {heatmaps && (
        <motion.div
          className="heatmap-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3>🔍 Explainability Heatmaps</h3>
          <div className="heatmap-grid">
            {heatmaps.original && (
              <div className="heatmap-card">
                <img src={`data:image/png;base64,${heatmaps.original}`} alt="Original scan" />
                <div className="heatmap-card-label">Original Scan</div>
              </div>
            )}
            {heatmaps.gradcam && (
              <div className="heatmap-card">
                <img src={`data:image/png;base64,${heatmaps.gradcam}`} alt="Grad-CAM overlay" />
                <div className="heatmap-card-label">Grad-CAM — Region of Interest</div>
              </div>
            )}
            {heatmaps.ig && (
              <div className="heatmap-card">
                <img src={`data:image/png;base64,${heatmaps.ig}`} alt="Integrated Gradients" />
                <div className="heatmap-card-label">Integrated Gradients — Pixel Attribution</div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* AI Explanation */}
      <motion.div
        className="ai-explanation"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <div className="ai-explanation-header">
          <Brain size={20} />
          {mode === 'practitioner' ? 'AI Clinical Analysis' : 'AI Explanation for You'}
        </div>
        
        {aiExplanation ? (
          <div className="ai-explanation-text">{aiExplanation}</div>
        ) : aiLoading ? (
          <div className="ai-loading">
            <div className="ai-loading-dots">
              <span /><span /><span />
            </div>
            AI is analysing the scan...
          </div>
        ) : (
          <button className="btn btn-primary" onClick={onGenerateExplanation}>
            <Brain size={16} /> Generate AI Explanation
          </button>
        )}
      </motion.div>
    </div>
  );
}
