import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Activity, LogOut, Menu } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Navbar.css';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handler);
    return () => window.removeEventListener('scroll', handler);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <motion.nav
      className={`navbar ${scrolled ? 'navbar-scrolled' : ''}`}
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <div className="navbar-inner">
        <Link to="/" className="navbar-logo">
          <div className="navbar-logo-icon">
            <Activity size={22} />
          </div>
          <div className="navbar-logo-text">
            Med<span>XAI</span>
          </div>
        </Link>

        <ul className="navbar-links">
          <li><Link to="/">Home</Link></li>
          <li><a href="/#how-it-works">How It Works</a></li>
          <li><a href="/#features">Features</a></li>
          {user && (
            <li>
              <Link to={user.role === 'practitioner' ? '/practitioner' : '/patient'}>
                Dashboard
              </Link>
            </li>
          )}
        </ul>

        <div className="navbar-actions">
          {user ? (
            <>
              <div className="navbar-user">
                <div className="navbar-user-avatar">
                  {user.name?.charAt(0).toUpperCase()}
                </div>
                <div>
                  <div className="navbar-user-name">{user.name}</div>
                  <div className="navbar-user-role">{user.role}</div>
                </div>
              </div>
              <button className="btn btn-small btn-outline" onClick={handleLogout}>
                <LogOut size={14} /> Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/auth?mode=signin" className="btn btn-secondary btn-small">Sign In</Link>
              <Link to="/auth?mode=signup" className="btn btn-primary btn-small">Get Started</Link>
            </>
          )}
        </div>
      </div>
    </motion.nav>
  );
}
