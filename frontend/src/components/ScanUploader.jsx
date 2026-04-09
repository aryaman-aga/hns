import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X } from 'lucide-react';
import './ScanUploader.css';

export default function ScanUploader({ onImageSelect, task, setTask, xaiMethod, setXaiMethod, showXai = true }) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      onImageSelect(file);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const removeImage = () => {
    setPreview(null);
    onImageSelect(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  return (
    <div>
      {/* Task Selector */}
      <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Select Disease Model
      </label>
      <div className="task-selector">
        <button
          className={`task-option ${task === 'pneumonia' ? 'selected' : ''}`}
          onClick={() => setTask('pneumonia')}
        >
          <span className="task-option-icon">🫁</span>
          Pneumonia
        </button>
        <button
          className={`task-option ${task === 'breast' ? 'selected' : ''}`}
          onClick={() => setTask('breast')}
        >
          <span className="task-option-icon">🔬</span>
          Breast Cancer
        </button>
      </div>

      {/* XAI Selector */}
      {showXai && (
        <>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Explainability Method
          </label>
          <div className="xai-selector">
            <button
              className={`xai-option ${xaiMethod === 'gradcam' ? 'selected' : ''}`}
              onClick={() => setXaiMethod('gradcam')}
            >
              🗺️ Grad-CAM
            </button>
            <button
              className={`xai-option ${xaiMethod === 'ig' ? 'selected' : ''}`}
              onClick={() => setXaiMethod('ig')}
            >
              🔬 Integrated Gradients
            </button>
            <button
              className={`xai-option ${xaiMethod === 'both' ? 'selected' : ''}`}
              onClick={() => setXaiMethod('both')}
            >
              🖼️ Both
            </button>
          </div>
        </>
      )}

      {/* Upload Area */}
      <div
        className={`scan-uploader ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/bmp,image/tiff"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        
        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              className="scan-preview-container"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <img src={preview} alt="Scan preview" className="scan-preview" />
              <button className="scan-preview-remove" onClick={(e) => { e.stopPropagation(); removeImage(); }}>
                <X size={14} />
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="scan-uploader-icon">
                <Upload size={28} />
              </div>
              <div className="scan-uploader-title">Drop your medical scan here</div>
              <div className="scan-uploader-hint">or click to browse — PNG, JPEG, BMP, TIFF</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
