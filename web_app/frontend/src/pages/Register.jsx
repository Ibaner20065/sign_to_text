import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Auth.css'

const Register = () => {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { register } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    setLoading(true)
    const result = await register(name, email, password)
    setLoading(false)

    if (result.success) {
      navigate('/login')
    } else {
      setError(result.error || 'Registration failed. Please try again.')
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-brand">
          <span className="auth-brand-icon">💙</span>
          <span className="auth-brand-name">AuraCare</span>
        </div>
        <h1>Create Account</h1>
        <p className="auth-subtitle">Join our inclusive healthcare platform</p>
        <form onSubmit={handleSubmit} id="register-form">
          <div className="form-group">
            <label className="label" htmlFor="register-name">Full Name</label>
            <input
              id="register-name"
              type="text"
              className="input"
              placeholder="John Doe"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              autoComplete="name"
            />
          </div>
          <div className="form-group">
            <label className="label" htmlFor="register-email">Email Address</label>
            <input
              id="register-email"
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
            <label className="label" htmlFor="register-password">Password</label>
            <input
              id="register-password"
              type="password"
              className="input"
              placeholder="Min. 6 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="new-password"
            />
          </div>
          {error && <div className="error-message" id="register-error">{error}</div>}
          <button type="submit" className="button button-primary" id="register-submit" disabled={loading}>
            {loading ? (
              <>
                <span className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }}></span>
                Creating account...
              </>
            ) : (
              'Create Account'
            )}
          </button>
        </form>
        <p className="auth-link">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </div>
    </div>
  )
}

export default Register
