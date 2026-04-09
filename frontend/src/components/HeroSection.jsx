import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Zap } from 'lucide-react';
import './HeroSection.css';

export default function HeroSection() {
  return (
    <section className="hero">
      <div className="hero-bg">
        <div className="hero-gradient" />
        <div className="hero-grid" />
      </div>

      <div className="hero-content">
        {/* Left side — Text */}
        <motion.div
          className="hero-text"
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <motion.div
            className="hero-badge"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <span className="hero-badge-dot" />
            Research-Grade AI Diagnostics
          </motion.div>

          <h1 className="hero-title">
            Dual-Granularity{' '}
            <span className="highlight">Explainability</span>{' '}
            for Medical Imaging
          </h1>

          <p className="hero-description">
            A multi-modal deep learning framework combining SE-ResNet-18 with 
            Grad-CAM and Integrated Gradients for transparent, trustworthy 
            medical image diagnosis at both region and pixel level.
          </p>

          <div className="hero-actions">
            <Link to="/auth?mode=signup" className="btn btn-primary btn-large">
              Get Started <ArrowRight size={18} />
            </Link>
            <a href="#how-it-works" className="btn btn-secondary btn-large">
              <Zap size={18} /> Learn How It Works
            </a>
          </div>

          <div className="hero-stats">
            <motion.div
              className="hero-stat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
            >
              <div className="hero-stat-value">93<span>%</span></div>
              <div className="hero-stat-label">AUC Score</div>
            </motion.div>
            <motion.div
              className="hero-stat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.0 }}
            >
              <div className="hero-stat-value">2<span>+</span></div>
              <div className="hero-stat-label">Disease Models</div>
            </motion.div>
            <motion.div
              className="hero-stat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
            >
              <div className="hero-stat-value">2<span>x</span></div>
              <div className="hero-stat-label">XAI Methods</div>
            </motion.div>
          </div>
        </motion.div>

        {/* Right side — Visual */}
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          <div className="hero-visual-ring hero-ring-1" />
          <div className="hero-visual-ring hero-ring-2" />
          <div className="hero-visual-ring hero-ring-3" />
          
          <motion.div
            className="hero-center-card"
            animate={{ y: [0, -12, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          >
            <div className="hero-center-icon">🧠</div>
            <div className="hero-center-label">SE-ResNet-18 + Dual XAI</div>
          </motion.div>

          <motion.div
            className="hero-floating-card hero-floating-1"
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
          >
            <span className="hero-floating-dot dot-green" />
            Grad-CAM Active
          </motion.div>

          <motion.div
            className="hero-floating-card hero-floating-2"
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
          >
            <span className="hero-floating-dot dot-blue" />
            Pixel Attribution
          </motion.div>

          <motion.div
            className="hero-floating-card hero-floating-3"
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut', delay: 1.5 }}
          >
            <span className="hero-floating-dot dot-orange" />
            AI Explanation
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
