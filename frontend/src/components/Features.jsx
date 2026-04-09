import { motion } from 'framer-motion';
import { Eye, Layers, Brain, Stethoscope, Shield, Cpu } from 'lucide-react';
import GlassCard from './GlassCard';
import './Features.css';

const features = [
  {
    icon: <Eye size={28} />,
    title: 'Grad-CAM Heatmaps',
    desc: 'Region-level activation mapping highlights the most influential areas in the medical scan for diagnosis.',
  },
  {
    icon: <Layers size={28} />,
    title: 'Pixel-Level Attribution',
    desc: 'Integrated Gradients with SmoothGrad² traces predictions to individual pixels for fine-grained insight.',
  },
  {
    icon: <Brain size={28} />,
    title: 'AI Clinical Explanations',
    desc: 'Local LLM (Ollama) generates human-readable explanations of model predictions in plain language.',
  },
  {
    icon: <Stethoscope size={28} />,
    title: 'Doctor & Patient Modes',
    desc: 'Practitioners get detailed severity analysis. Patients receive clear guidance on whether to see a doctor.',
  },
  {
    icon: <Shield size={28} />,
    title: 'Multi-Disease Support',
    desc: 'Currently supports pneumonia detection and breast cancer screening, with NIH Chest-14 coming soon.',
  },
  {
    icon: <Cpu size={28} />,
    title: 'SE-ResNet-18 Architecture',
    desc: 'Squeeze-Excitation attention blocks enhance channel-wise feature recalibration for better accuracy.',
  },
];

export default function Features() {
  return (
    <section id="features" className="section features-section">
      <div className="container">
        <motion.h2
          className="section-title"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          Powerful Features
        </motion.h2>
        <motion.p
          className="section-subtitle"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
        >
          Enterprise-grade explainability tools built for clinical trust
        </motion.p>

        <div className="features-grid">
          {features.map((f, i) => (
            <GlassCard
              key={i}
              icon={f.icon}
              title={f.title}
              description={f.desc}
              delay={i * 0.1}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
