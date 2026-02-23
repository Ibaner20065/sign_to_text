import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import AuthModal from '../components/AuthModal'
import './Home.css'

const Home = () => {
    const { isAuthenticated } = useAuth()
    const navigate = useNavigate()
    const [showAuthModal, setShowAuthModal] = useState(false)
    const [authMode, setAuthMode] = useState('login')
    const [currentImageIndex, setCurrentImageIndex] = useState(0)

    // Rotate hero illustrations
    const heroIllustrations = [
        { emoji: '🤟', label: 'Sign Language Support', color: '#4F46E5' },
        { emoji: '👩‍⚕️', label: 'Doctor Communication', color: '#0EA5A4' },
        { emoji: '🏥', label: 'Accessible Healthcare', color: '#6366f1' },
    ]

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentImageIndex((prev) => (prev + 1) % heroIllustrations.length)
        }, 4000)
        return () => clearInterval(interval)
    }, [])

    const openAuth = (mode) => {
        setAuthMode(mode)
        setShowAuthModal(true)
    }

    const features = [
        {
            icon: '🔍',
            title: 'Find a Doctor',
            description: 'AI-assisted search with sign-language support. Find specialists who understand your needs.',
            link: isAuthenticated ? '/hospitals' : null,
            action: isAuthenticated ? null : () => openAuth('login'),
            badge: '🤟 Sign Language Available',
        },
        {
            icon: '📍',
            title: 'Deaf-Friendly Locations',
            description: 'Clinics and hospitals trained in accessibility. Video call interpreters on demand.',
            link: isAuthenticated ? '/hospitals' : null,
            action: isAuthenticated ? null : () => openAuth('login'),
            badge: '♿ Accessibility Verified',
        },
        {
            icon: '📋',
            title: 'Patient Portal',
            description: 'Appointments, translation services, medical records — all in one accessible dashboard.',
            link: isAuthenticated ? '/communication' : null,
            action: isAuthenticated ? null : () => openAuth('login'),
            badge: '🔒 Secure Access',
        },
    ]

    const accessibilityFeatures = [
        { icon: '🤟', title: 'Sign Language Recognition', desc: 'Real-time AI-powered gesture detection' },
        { icon: '🗣️', title: 'Speech-to-Text', desc: 'Instant voice transcription in appointments' },
        { icon: '📹', title: 'Video Call Support', desc: 'Remote interpreter connections' },
        { icon: '📄', title: 'Document Scanner', desc: 'AI scam detection on medical bills' },
        { icon: '🚑', title: 'Emergency Access', desc: 'No-login ambulance & hospital finder' },
        { icon: '🌐', title: 'Multi-Language', desc: 'ASL & ISL support built-in' },
    ]

    return (
        <div className="home-page">
            {/* ═══ HERO SECTION ═══ */}
            <section className="hero" id="hero">
                <div className="hero-bg-shapes">
                    <div className="hero-shape hero-shape-1"></div>
                    <div className="hero-shape hero-shape-2"></div>
                    <div className="hero-shape hero-shape-3"></div>
                </div>

                <div className="hero-container">
                    <div className="hero-text">
                        <div className="hero-badge">
                            <span className="hero-badge-dot"></span>
                            AI-Powered Inclusive Healthcare
                        </div>
                        <h1 className="hero-headline">
                            Inclusive Healthcare
                            <br />
                            <span className="hero-headline-gradient">for Everyone</span>
                        </h1>
                        <p className="hero-subtext">
                            Breaking communication barriers between deaf patients and healthcare providers with
                            AI-powered sign language recognition, speech-to-text, and accessible medical services.
                        </p>
                        <div className="hero-cta-group">
                            {isAuthenticated ? (
                                <>
                                    <Link to="/hospitals" className="hero-cta hero-cta-primary" id="cta-find-doctor">
                                        <span>🔍</span> Find Doctor
                                    </Link>
                                    <Link to="/communication" className="hero-cta hero-cta-secondary" id="cta-communicate">
                                        <span>🤟</span> Start Communicating
                                    </Link>
                                </>
                            ) : (
                                <>
                                    <button onClick={() => openAuth('register')} className="hero-cta hero-cta-primary" id="cta-get-started">
                                        <span>✨</span> Get Started Free
                                    </button>
                                    <button onClick={() => openAuth('login')} className="hero-cta hero-cta-secondary" id="cta-sign-in">
                                        <span>→</span> Sign In
                                    </button>
                                </>
                            )}
                        </div>
                        <div className="hero-trust">
                            <div className="hero-trust-avatars">
                                <div className="trust-avatar" style={{ background: '#4F46E5' }}>A</div>
                                <div className="trust-avatar" style={{ background: '#0EA5A4' }}>B</div>
                                <div className="trust-avatar" style={{ background: '#6366f1' }}>C</div>
                                <div className="trust-avatar" style={{ background: '#818cf8' }}>D</div>
                            </div>
                            <span className="hero-trust-text">Trusted by healthcare providers & patients alike</span>
                        </div>
                    </div>

                    <div className="hero-visual">
                        <div className="hero-visual-card">
                            <div className="hero-illustration">
                                <div className="hero-illustration-circle">
                                    {heroIllustrations.map((item, i) => (
                                        <div
                                            key={i}
                                            className={`hero-illustration-item ${i === currentImageIndex ? 'active' : ''}`}
                                        >
                                            <span className="hero-illustration-emoji">{item.emoji}</span>
                                            <span className="hero-illustration-label">{item.label}</span>
                                        </div>
                                    ))}
                                </div>
                                {/* Decorative elements */}
                                <div className="hero-floating-badge hero-fb-1">
                                    <span>🤟</span> Sign Language Available
                                </div>
                                <div className="hero-floating-badge hero-fb-2">
                                    <span>📹</span> Video Call
                                </div>
                                <div className="hero-floating-badge hero-fb-3">
                                    <span>♿</span> WCAG Accessible
                                </div>
                            </div>
                            {/* Status indicator */}
                            <div className="hero-status">
                                <span className="hero-status-dot"></span>
                                <span>AI Engine Active — Ready to assist</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ═══ FEATURE CARDS ═══ */}
            <section className="features" id="services">
                <div className="features-container">
                    <div className="features-header">
                        <h2 className="features-title">How We Help</h2>
                        <p className="features-subtitle">
                            Three powerful ways to access inclusive healthcare
                        </p>
                    </div>
                    <div className="features-grid">
                        {features.map((feature, index) => (
                            <div
                                key={index}
                                className="feature-card"
                                style={{ animationDelay: `${index * 0.15}s` }}
                                onClick={feature.action || undefined}
                            >
                                {feature.link ? (
                                    <Link to={feature.link} className="feature-card-link">
                                        <FeatureCardContent feature={feature} />
                                    </Link>
                                ) : (
                                    <div className="feature-card-link" style={{ cursor: feature.action ? 'pointer' : 'default' }}>
                                        <FeatureCardContent feature={feature} />
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ═══ ACCESSIBILITY SECTION ═══ */}
            <section className="accessibility-section" id="accessibility">
                <div className="accessibility-container">
                    <div className="accessibility-header">
                        <div className="accessibility-badge">♿ Built for Everyone</div>
                        <h2 className="accessibility-title">Accessibility at Our Core</h2>
                        <p className="accessibility-subtitle">
                            Every feature designed with deaf and hard-of-hearing users in mind
                        </p>
                    </div>
                    <div className="accessibility-grid">
                        {accessibilityFeatures.map((item, index) => (
                            <div
                                key={index}
                                className="accessibility-card"
                                style={{ animationDelay: `${index * 0.1}s` }}
                            >
                                <span className="accessibility-card-icon">{item.icon}</span>
                                <h3>{item.title}</h3>
                                <p>{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ═══ EMERGENCY BANNER ═══ */}
            <section className="emergency-banner">
                <div className="emergency-container">
                    <div className="emergency-content">
                        <span className="emergency-icon">🚨</span>
                        <div>
                            <h3>Emergency? No account needed.</h3>
                            <p>Access ambulance tracking and hospital finder instantly</p>
                        </div>
                    </div>
                    <Link to="/emergency" className="emergency-cta" id="cta-emergency">
                        Emergency Access →
                    </Link>
                </div>
            </section>

            {/* ═══ FOOTER ═══ */}
            <footer className="home-footer">
                <div className="footer-container">
                    <div className="footer-brand">
                        <span className="footer-logo">❤️</span>
                        <span className="footer-name">AuraCare</span>
                        <p className="footer-tagline">Inclusive healthcare for everyone</p>
                    </div>
                    <div className="footer-links">
                        <div className="footer-col">
                            <h4>Services</h4>
                            <a href="#services">Sign Language</a>
                            <a href="#services">Speech Recognition</a>
                            <a href="#services">Document Scanner</a>
                        </div>
                        <div className="footer-col">
                            <h4>Accessibility</h4>
                            <a href="#accessibility">WCAG Compliant</a>
                            <a href="#accessibility">ASL Support</a>
                            <a href="#accessibility">ISL Support</a>
                        </div>
                        <div className="footer-col">
                            <h4>Quick Access</h4>
                            <Link to="/emergency">Emergency</Link>
                            <a href="#services">Hospitals</a>
                            <a href="#hero">Get Started</a>
                        </div>
                    </div>
                    <div className="footer-bottom">
                        <p>© 2025 AuraCare. Built with ❤️ for inclusive healthcare.</p>
                        <div className="footer-accessibility-tags">
                            <span className="footer-tag">♿ WCAG 2.1</span>
                            <span className="footer-tag">🤟 ASL</span>
                            <span className="footer-tag">🔒 HIPAA-Ready</span>
                        </div>
                    </div>
                </div>
            </footer>

            {/* ═══ AUTH MODAL ═══ */}
            {showAuthModal && (
                <AuthModal
                    mode={authMode}
                    onClose={() => setShowAuthModal(false)}
                    onSwitchMode={(mode) => setAuthMode(mode)}
                />
            )}
        </div>
    )
}

/* Feature card inner content */
const FeatureCardContent = ({ feature }) => (
    <>
        <div className="feature-card-icon">{feature.icon}</div>
        <h3 className="feature-card-title">{feature.title}</h3>
        <p className="feature-card-desc">{feature.description}</p>
        <div className="feature-card-badge">{feature.badge}</div>
        <div className="feature-card-arrow">→</div>
    </>
)

export default Home
