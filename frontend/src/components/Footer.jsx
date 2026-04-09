import { Activity, GraduationCap } from 'lucide-react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-logo">
          <div className="footer-logo-icon">
            <Activity size={18} />
          </div>
          <div className="footer-logo-text">Med<span>XAI</span></div>
        </div>

        <div className="footer-info">
          <div className="footer-paper">
            &ldquo;Enhancing Clinical Trust: A Multi-Modal Deep Learning Framework
            with Dual-Granularity Explainability for Medical Image Diagnosis&rdquo;
          </div>
          <div className="footer-copy">
            &copy; {new Date().getFullYear()} Aryaman Agarwal, Kanik Chawla, Sparsh Kalia — NSUT
          </div>
        </div>

        <div className="footer-nsut">
          <GraduationCap size={16} />
          NSUT, Delhi
        </div>
      </div>
    </footer>
  );
}
