import { motion } from 'framer-motion';
import './GlassCard.css';

export default function GlassCard({ icon, title, description, delay = 0, children, className = '' }) {
  return (
    <motion.div
      className={`glass-card ${className}`}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.6, delay }}
      animate={{
        y: [0, -8, 0],
      }}
      whileHover={{ scale: 1.02 }}
    >
      {icon && <div className="glass-card-icon">{icon}</div>}
      {title && <h3>{title}</h3>}
      {description && <p>{description}</p>}
      {children}
    </motion.div>
  );
}
