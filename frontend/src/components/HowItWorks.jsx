import { motion } from 'framer-motion';
import { Upload, Brain, FileCheck } from 'lucide-react';
import './HowItWorks.css';

const steps = [
  {
    num: 1,
    icon: '📤',
    title: 'Upload Medical Scan',
    desc: 'Upload a chest X-ray or breast ultrasound image. Our system accepts any resolution and converts it internally.',
  },
  {
    num: 2,
    icon: '🧠',
    title: 'AI Analysis & XAI',
    desc: 'SE-ResNet-18 classifies the image while Grad-CAM highlights regions and Integrated Gradients maps pixel-level attributions.',
  },
  {
    num: 3,
    icon: '📋',
    title: 'Get Expert Report',
    desc: 'Receive severity prediction, visual heatmaps, and AI-generated clinical explanations — all in one intuitive report.',
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="section how-it-works">
      <div className="container">
        <motion.h2
          className="section-title"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          How It Works
        </motion.h2>
        <motion.p
          className="section-subtitle"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
        >
          Three simple steps from scan upload to AI-powered diagnostic insight
        </motion.p>

        <div className="how-it-works-grid">
          {steps.map((step, i) => (
            <motion.div
              key={step.num}
              className="step-card"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.2 }}
            >
              <motion.div
                className="step-number"
                animate={{ y: [0, -6, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: i * 0.3 }}
              >
                {step.num}
              </motion.div>
              <div className="step-icon-wrapper">{step.icon}</div>
              <h3 className="step-title">{step.title}</h3>
              <p className="step-desc">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
