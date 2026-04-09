import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Stethoscope, User } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Auth.css';

export default function Auth() {
  const [searchParams] = useSearchParams();
  const initialMode = searchParams.get('mode') || 'signin';
  const initialRole = searchParams.get('role') || 'practitioner';

  const [mode, setMode] = useState(initialMode);
  const [role, setRole] = useState(initialRole);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [specialization, setSpecialization] = useState('');
  const [licenseId, setLicenseId] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login, signup, user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      navigate(user.role === 'practitioner' ? '/practitioner' : '/patient');
    }
  }, [user, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (mode === 'signin') {
        await login(email, password);
      } else {
        await signup({
          name,
          email,
          password,
          role,
          ...(role === 'practitioner' ? { specialization, licenseId } : {}),
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-bg" />
      <div className="bg-decoration">
        <div className="bg-blob bg-blob-1" />
        <div className="bg-blob bg-blob-2" />
      </div>

      <motion.div
        className="auth-container"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="auth-card">
          <div className="auth-header">
            <h1>{mode === 'signin' ? 'Welcome Back' : 'Create Account'}</h1>
            <p>{mode === 'signin' ? 'Sign in to access your dashboard' : 'Join our AI-powered diagnostic platform'}</p>
          </div>

          {/* Role Toggle */}
          <div className="role-toggle">
            <div className={`role-toggle-slider ${role === 'practitioner' ? 'left' : 'right'}`} />
            <button
              className={`role-toggle-btn ${role === 'practitioner' ? 'active' : ''}`}
              onClick={() => setRole('practitioner')}
              type="button"
            >
              <Stethoscope size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Practitioner
            </button>
            <button
              className={`role-toggle-btn ${role === 'patient' ? 'active' : ''}`}
              onClick={() => setRole('patient')}
              type="button"
            >
              <User size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Patient
            </button>
          </div>

          {error && <div className="auth-error">{error}</div>}

          <form className="auth-form" onSubmit={handleSubmit}>
            {mode === 'signup' && (
              <div className="input-group">
                <label>Full Name</label>
                <input
                  className="input-field"
                  type="text"
                  placeholder="Dr. Jane Smith"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            )}

            <div className="input-group">
              <label>Email</label>
              <input
                className="input-field"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="input-group">
              <label>Password</label>
              <input
                className="input-field"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
              />
            </div>

            {mode === 'signup' && role === 'practitioner' && (
              <>
                <div className="input-group">
                  <label>Specialization</label>
                  <input
                    className="input-field"
                    type="text"
                    placeholder="Radiology, Pulmonology, etc."
                    value={specialization}
                    onChange={(e) => setSpecialization(e.target.value)}
                  />
                </div>
                <div className="input-group">
                  <label>Medical License ID</label>
                  <input
                    className="input-field"
                    type="text"
                    placeholder="License / Registration Number"
                    value={licenseId}
                    onChange={(e) => setLicenseId(e.target.value)}
                  />
                </div>
              </>
            )}

            <button
              className="btn btn-primary btn-large"
              type="submit"
              disabled={loading}
              style={{ width: '100%', marginTop: '8px' }}
            >
              {loading ? 'Please wait...' : (mode === 'signin' ? 'Sign In' : 'Create Account')}
            </button>
          </form>

          <div className="mode-toggle">
            {mode === 'signin' ? (
              <>
                Don't have an account?{' '}
                <button onClick={() => setMode('signup')}>Sign Up</button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button onClick={() => setMode('signin')}>Sign In</button>
              </>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
