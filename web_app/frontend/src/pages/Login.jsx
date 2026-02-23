import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Activity, Siren } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import './Auth.css'

const Login = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [rememberMe, setRememberMe] = useState(true)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    const result = await login(email, password, rememberMe)
    setLoading(false)

    if (result.success) {
      navigate('/communication')
    } else {
      setError(result.error || 'Login failed. Please check your credentials.')
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-brand">
          <div className="auth-brand-logo">
            <Activity size={32} />
          </div>
          <span className="auth-brand-name">AuraCare</span>
        </div>
        <h1>Welcome Back</h1>
        <p className="auth-subtitle">Sign in to your healthcare dashboard</p>
        <form onSubmit={handleSubmit} id="login-form">
          <div className="form-group">
            <label className="label" htmlFor="login-email">Email Address</label>
            <input
              id="login-email"
              type="email"
              className="input"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>
          <div className="form-group">
            <label className="label" htmlFor="login-password">Password</label>
            <input
              id="login-password"
              type="password"
              className="input"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </div>
          <label className="form-checkbox">
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
            />
            <span>Remember me (Stay signed in)</span>
          </label>
          {error && <div className="error-message" id="login-error">{error}</div>}
          <button type="submit" className="button button-primary" id="login-submit" disabled={loading}>
            {loading ? (
              <>
                <span className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }}></span>
                Signing in...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>
        <div className="auth-divider">or</div>
        <Link to="/emergency" className="button button-danger" id="emergency-access" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', width: '100%', textAlign: 'center', fontSize: '1rem', fontWeight: 600, padding: '14px 20px', textDecoration: 'none' }}>
          <Siren size={20} /> Emergency Access
        </Link>
        <p style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>
          Access hospitals & ambulance without an account
        </p>
        <p className="auth-link">
          Don't have an account? <Link to="/register">Create one</Link>
        </p>
      </div>
    </div>
  )
}

export default Login
