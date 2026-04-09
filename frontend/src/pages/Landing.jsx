import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, UserCheck } from 'lucide-react';
import HeroSection from '../components/HeroSection';
import HowItWorks from '../components/HowItWorks';
import Features from '../components/Features';
import './Landing.css';

const archText = `Input (1×224×224 grayscale)
       │
  Modified Conv1  (7×7, s=2, 64ch)
       │
  Layer1 — BasicBlock×2 + SE  →  64ch,  56×56
  Layer2 — BasicBlock×2 + SE  → 128ch,  28×28
  Layer3 — BasicBlock×2 + SE  → 256ch,  14×14
  Layer4 — BasicBlock×2 + SE  → 512ch,   7×7
       │              ↑ Grad-CAM target
  AdaptiveAvgPool2d → (512-d)
       │
  Dropout → FC(128) → ReLU → FC(2)
       │
  Softmax → Prediction`;

export default function Landing() {
  return (
    <>
      {/* Background Decoration */}
      <div className="bg-decoration">
        <div className="bg-blob bg-blob-1" />
        <div className="bg-blob bg-blob-2" />
        <div className="bg-blob bg-blob-3" />
      </div>

      <HeroSection />
      <HowItWorks />
      <Features />

      {/* About the Model */}
      <section className="section landing-about">
        <div className="container">
          <motion.h2
            className="section-title"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            The Architecture
          </motion.h2>
          <motion.p
            className="section-subtitle"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            SE-ResNet-18 with Squeeze-Excitation attention for channel recalibration
          </motion.p>

          <div className="about-grid">
            <motion.div
              className="about-arch"
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              {archText}
            </motion.div>

            <motion.div
              className="about-text"
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <h2>Research-Grade Precision</h2>
              <p>
                Our SE-ResNet-18 architecture enhances the standard ResNet with 
                Squeeze-Excitation blocks after every residual layer. This allows 
                the network to dynamically recalibrate channel-wise feature 
                responses, focusing on the most informative features.
              </p>
              <p>
                Combined with Cosine Annealing LR, Mixup augmentation, and 
                label smoothing, the model achieves state-of-the-art performance 
                on medical imaging benchmarks.
              </p>

              <div className="about-stats-row">
                <motion.div
                  className="about-stat-card"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <div className="about-stat-val">11.5M</div>
                  <div className="about-stat-lbl">Parameters</div>
                </motion.div>
                <motion.div
                  className="about-stat-card"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
                >
                  <div className="about-stat-val">93.4%</div>
                  <div className="about-stat-lbl">AUC Score</div>
                </motion.div>
                <motion.div
                  className="about-stat-card"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
                >
                  <div className="about-stat-val">~1s</div>
                  <div className="about-stat-lbl">Grad-CAM Speed</div>
                </motion.div>
                <motion.div
                  className="about-stat-card"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 1.5 }}
                >
                  <div className="about-stat-val">224²</div>
                  <div className="about-stat-lbl">Input Resolution</div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section landing-cta">
        <div className="container">
          <motion.div
            className="cta-card"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            animate={{ y: [0, -8, 0] }}
          >
            <h2>Ready to Get Started?</h2>
            <p>
              Whether you're a medical practitioner seeking detailed diagnostic 
              analysis or a patient looking for guidance — our AI is here to help.
            </p>
            <div className="cta-actions">
              <Link to="/auth?mode=signup&role=practitioner" className="btn btn-primary btn-large">
                <UserCheck size={18} /> I'm a Practitioner
              </Link>
              <Link to="/auth?mode=signup&role=patient" className="btn btn-outline btn-large">
                I'm a Patient <ArrowRight size={18} />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </>
  );
}
