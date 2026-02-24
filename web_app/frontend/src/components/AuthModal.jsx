import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './AuthModal.css'

const AuthModal = ({ mode = 'login', onClose, onSwitchMode }) => {
    const [currentMode, setCurrentMode] = useState(mode)
    const [name, setName] = useState('')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)
    const [isHumanVerified, setIsHumanVerified] = useState(false)
    const { login, register } = useAuth()
    const navigate = useNavigate()

    useEffect(() => {
        setCurrentMode(mode)
    }, [mode])

    // Close on Escape key
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') onClose()
        }
        document.addEventListener('keydown', handleEscape)
        return () => document.removeEventListener('keydown', handleEscape)
    }, [onClose])

    // Lock body scroll when modal is open
    useEffect(() => {
        document.body.style.overflow = 'hidden'
        return () => { document.body.style.overflow = '' }
    }, [])

    const switchMode = (newMode) => {
        setCurrentMode(newMode)
        setError('')
        setEmail('')
        setPassword('')
        setName('')
        setIsHumanVerified(false)
        onSwitchMode?.(newMode)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        setLoading(true)

        if (currentMode === 'login') {
            const result = await login(email, password)
            setLoading(false)
            if (result.success) {
                onClose()
                navigate('/communication')
            } else {
                setError(result.error || 'Login failed. Please check your credentials.')
            }
        } else {
            if (password.length < 8) {
                setError('Password must be at least 8 characters')
                setLoading(false)
                return
            }
            const result = await register(name, email, password)
            setLoading(false)
            if (result.success) {
                switchMode('login')
                setError('')
            } else {
                setError(result.error || 'Registration failed. Please try again.')
            }
        }
    }

    return (
        <div className="auth-modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label={currentMode === 'login' ? 'Sign In' : 'Create Account'}>
            <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
                {/* Close button */}
                <button className="auth-modal-close" onClick={onClose} aria-label="Close">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M18 6L6 18M6 6l12 12" />
                    </svg>
                </button>

                {/* Brand */}
                <div className="auth-modal-brand">
                    <span className="auth-modal-brand-name">AuraCare</span>
                </div>

                {/* Tabs */}
                <div className="auth-modal-tabs">
                    <button
                        className={`auth-modal-tab ${currentMode === 'login' ? 'active' : ''}`}
                        onClick={() => switchMode('login')}
                    >
                        Sign In
                    </button>
                    <button
                        className={`auth-modal-tab ${currentMode === 'register' ? 'active' : ''}`}
                        onClick={() => switchMode('register')}
                    >
                        Create Account
                    </button>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="auth-modal-form">
                    {currentMode === 'register' && (
                        <div className="auth-modal-field">
                            <label htmlFor="auth-name">Full Name</label>
                            <input
                                id="auth-name"
                                type="text"
                                placeholder="Your full name"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                required
                                autoComplete="name"
                            />
                        </div>
                    )}
                    <div className="auth-modal-field">
                        <label htmlFor="auth-email">Email Address</label>
                        <input
                            id="auth-email"
                            type="email"
                            placeholder="you@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            autoComplete="email"
                        />
                    </div>
                    <div className="auth-modal-field">
                        <label htmlFor="auth-password">Password</label>
                        <input
                            id="auth-password"
                            type="password"
                            placeholder={currentMode === 'register' ? 'Min. 8 characters' : '••••••••'}
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            autoComplete={currentMode === 'login' ? 'current-password' : 'new-password'}
                        />
                    </div>

                    <div className="auth-modal-verification">
                        <input
                            type="checkbox"
                            id="human-verify"
                            className="auth-verification-checkbox"
                            checked={isHumanVerified}
                            onChange={(e) => setIsHumanVerified(e.target.checked)}
                        />
                        <label htmlFor="human-verify" className="auth-verification-label">
                            I verify that I am a human healthcare professional
                        </label>
                    </div>

                    {error && <div className="auth-modal-error">{error}</div>}

                    <button type="submit" className="auth-modal-submit" disabled={loading || !isHumanVerified}>
                        {loading ? (
                            <span className="auth-modal-loading">
                                <span className="auth-spinner"></span>
                                {currentMode === 'login' ? 'Signing in...' : 'Creating account...'}
                            </span>
                        ) : (
                            currentMode === 'login' ? 'Sign In' : 'Create Account'
                        )}
                    </button>
                </form>

                <p className="auth-modal-switch">
                    {currentMode === 'login' ? (
                        <>Don't have an account? <button onClick={() => switchMode('register')}>Create one</button></>
                    ) : (
                        <>Already have an account? <button onClick={() => switchMode('login')}>Sign in</button></>
                    )}
                </p>
            </div>
        </div>
    )
}

export default AuthModal
