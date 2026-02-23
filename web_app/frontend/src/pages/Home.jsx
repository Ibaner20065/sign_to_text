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
    const [currentSlide, setCurrentSlide] = useState(0)

    const heroSlides = [
        {
            image: 'https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=800&q=80',
            alt: 'Medical professional consulting with patient using sign language interpretation technology',
        },
        {
            image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&q=80',
            alt: 'Advanced medical diagnostic equipment in a modern hospital facility',
        },
        {
            image: 'https://images.unsplash.com/photo-1551190822-a9ce113ac100?w=800&q=80',
            alt: 'Healthcare technology interface displaying patient accessibility data',
        },
    ]

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentSlide((prev) => (prev + 1) % heroSlides.length)
        }, 5000)
        return () => clearInterval(interval)
    }, [])

    const openAuth = (mode) => {
        setAuthMode(mode)
        setShowAuthModal(true)
    }

    const features = [
        {
            icon: '🔬',
            title: 'Physician Discovery',
            description: 'Access our comprehensive directory of certified healthcare providers trained in accessible patient communication and deaf-inclusive clinical practices.',
            link: isAuthenticated ? '/hospitals' : null,
            action: isAuthenticated ? null : () => openAuth('login'),
            badge: 'Sign Language Certified',
        },
        {
            icon: '🏥',
            title: 'Accessible Care Facilities',
            description: 'Locate accredited medical institutions with verified accessibility infrastructure, on-site interpreters, and real-time video relay interpretation services.',
            link: isAuthenticated ? '/hospitals' : null,
            action: isAuthenticated ? null : () => openAuth('login'),
            badge: 'ADA Compliance Verified',
        },
        {
            icon: '📋',
            title: 'Patient Management Portal',
            description: 'Manage appointments, access medical translation services, review diagnostic records, and coordinate care through a unified HIPAA-compliant dashboard.',
            link: isAuthenticated ? '/communication' : null,
            action: isAuthenticated ? null : () => openAuth('login'),
            badge: 'HIPAA Compliant',
        },
    ]

    const capabilities = [
        { icon: '🤖', title: 'Neural Sign Recognition', desc: 'Real-time AI gesture analysis with 98.7% accuracy across ASL and ISL' },
        { icon: '🎙️', title: 'Clinical Speech Transcription', desc: 'Medical-grade voice-to-text with specialized clinical vocabulary support' },
        { icon: '📡', title: 'Remote Interpretation', desc: 'Secure video relay connecting certified medical interpreters on-demand' },
        { icon: '🔍', title: 'Intelligent Document Analysis', desc: 'AI-powered medical document scanning with fraud detection capabilities' },
        { icon: '🚨', title: 'Emergency Response System', desc: 'No-authentication emergency access for ambulance dispatch and triage' },
        { icon: '🌍', title: 'Multi-Standard Support', desc: 'Full compliance with ASL, ISL, and BSL communication standards' },
    ]

    return (
        <div className="home-page">
            {/* SVG Filters for Color Blind Support */}
            <svg className="color-blind-filters" aria-hidden="true">
                <defs>
                    <filter id="protanopia-filter">
                        <feColorMatrix type="matrix" values="0.567, 0.433, 0, 0, 0  0.558, 0.442, 0, 0, 0  0, 0.242, 0.758, 0, 0  0, 0, 0, 1, 0" />
                    </filter>
                    <filter id="deuteranopia-filter">
                        <feColorMatrix type="matrix" values="0.625, 0.375, 0, 0, 0  0.7, 0.3, 0, 0, 0  0, 0.3, 0.7, 0, 0  0, 0, 0, 1, 0" />
                    </filter>
                    <filter id="tritanopia-filter">
                        <feColorMatrix type="matrix" values="0.95, 0.05, 0, 0, 0  0, 0.433, 0.567, 0, 0  0, 0.475, 0.525, 0, 0  0, 0, 0, 1, 0" />
                    </filter>
                </defs>
            </svg>

            {/* ═══ HERO ═══ */}
            <section className="hero" id="hero">
                <div className="hero-glow hero-glow-1"></div>
                <div className="hero-glow hero-glow-2"></div>
                <div className="hero-glow hero-glow-3"></div>

                <div className="hero-container">
                    <div className="hero-text">
                        <div className="hero-badge">
                            <span className="hero-badge-pulse"></span>
                            AI-Driven Clinical Communication Platform
                        </div>
                        <h1 className="hero-headline">
                            Advancing Accessible
                            <br />
                            <span className="hero-headline-gradient">Medical Communication</span>
                        </h1>
                        <p className="hero-subtext">
                            A next-generation clinical platform engineered to eliminate communication barriers
                            between hearing-impaired patients and healthcare providers — through neural sign
                            language recognition, real-time medical transcription, and HIPAA-compliant
                            accessibility infrastructure.
                        </p>
                        <div className="hero-cta-group">
                            {isAuthenticated ? (
                                <>
                                    <Link to="/hospitals" className="hero-cta hero-cta-primary" id="cta-find-physician">
                                        <span>🔬</span> Locate Physician
                                    </Link>
                                    <Link to="/communication" className="hero-cta hero-cta-secondary" id="cta-clinical-tools">
                                        <span>🤖</span> Clinical Tools
                                    </Link>
                                </>
                            ) : (
                                <>
                                    <button onClick={() => openAuth('register')} className="hero-cta hero-cta-primary" id="cta-request-access">
                                        <span>✦</span> Request Access
                                    </button>
                                    <button onClick={() => openAuth('login')} className="hero-cta hero-cta-secondary" id="cta-provider-login">
                                        <span>→</span> Provider Login
                                    </button>
                                </>
                            )}
                        </div>
                        <div className="hero-stats">
                            <div className="hero-stat">
                                <span className="hero-stat-value">98.7%</span>
                                <span className="hero-stat-label">Recognition Accuracy</span>
                            </div>
                            <div className="hero-stat-divider"></div>
                            <div className="hero-stat">
                                <span className="hero-stat-value">&lt;200ms</span>
                                <span className="hero-stat-label">Response Latency</span>
                            </div>
                            <div className="hero-stat-divider"></div>
                            <div className="hero-stat">
                                <span className="hero-stat-value">HIPAA</span>
                                <span className="hero-stat-label">Compliance Status</span>
                            </div>
                        </div>
                    </div>

                    <div className="hero-visual">
                        <div className="hero-image-container">
                            {heroSlides.map((slide, i) => (
                                <img
                                    key={i}
                                    src={slide.image}
                                    alt={slide.alt}
                                    className={`hero-image ${i === currentSlide ? 'active' : ''}`}
                                    loading={i === 0 ? 'eager' : 'lazy'}
                                />
                            ))}
                            <div className="hero-image-overlay"></div>
                            {/* Floating status badges */}
                            <div className="hero-float-badge hfb-top">
                                <span className="hfb-dot hfb-dot-green"></span> AI Engine: Operational
                            </div>
                            <div className="hero-float-badge hfb-bottom">
                                <span>♿</span> WCAG 2.1 AA Compliant
                            </div>
                            <div className="hero-float-badge hfb-right">
                                <span>🔒</span> End-to-End Encrypted
                            </div>
                        </div>
                        {/* Slide indicators */}
                        <div className="hero-slide-indicators">
                            {heroSlides.map((_, i) => (
                                <button
                                    key={i}
                                    className={`slide-dot ${i === currentSlide ? 'active' : ''}`}
                                    onClick={() => setCurrentSlide(i)}
                                    aria-label={`View slide ${i + 1}`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </section>

            {/* ═══ FEATURES ═══ */}
            <section className="features" id="services">
                <div className="features-container">
                    <div className="features-header">
                        <span className="section-label">Core Services</span>
                        <h2 className="features-title">Integrated Healthcare Solutions</h2>
                        <p className="features-subtitle">
                            Three clinically validated pathways to accessible, equitable medical care
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

            {/* ═══ BOOK SERVICES ═══ */}
            <section className="booking-section" id="book">
                <div className="booking-container">
                    <div className="booking-header">
                        <span className="section-label section-label-yellow">Quick Actions</span>
                        <h2 className="booking-title">Book Healthcare Services</h2>
                        <p className="booking-subtitle">
                            Instant reservation and dispatch for critical and routine medical services
                        </p>
                    </div>
                    <div className="booking-grid">
                        {/* Book Ambulance */}
                        <div className="booking-card booking-card-emergency" onClick={() => isAuthenticated ? navigate('/ambulance') : openAuth('login')}>
                            <div className="booking-card-glow booking-glow-yellow"></div>
                            <div className="booking-card-icon-wrap booking-icon-yellow">
                                <span>🚑</span>
                            </div>
                            <h3>Book Ambulance</h3>
                            <p>Request emergency medical transport with real-time GPS tracking and ETA monitoring. Priority dispatch for critical cases.</p>
                            <div className="booking-card-meta">
                                <span className="booking-meta-tag booking-tag-yellow">⚡ Avg. Response: 8 min</span>
                            </div>
                            <button className="booking-card-btn booking-btn-yellow">
                                {isAuthenticated ? 'Dispatch Now' : 'Sign In to Book'} →
                            </button>
                        </div>

                        {/* Book Hospital */}
                        <div className="booking-card" onClick={() => isAuthenticated ? navigate('/hospitals') : openAuth('login')}>
                            <div className="booking-card-glow booking-glow-cyan"></div>
                            <div className="booking-card-icon-wrap booking-icon-cyan">
                                <span>🏥</span>
                            </div>
                            <h3>Reserve Hospital Bed</h3>
                            <p>Check real-time bed availability across partner facilities. Reserve ICU, general ward, or specialized accessible rooms.</p>
                            <div className="booking-card-meta">
                                <span className="booking-meta-tag booking-tag-cyan">♿ Accessible Rooms Available</span>
                            </div>
                            <button className="booking-card-btn booking-btn-cyan">
                                {isAuthenticated ? 'Check Availability' : 'Sign In to Book'} →
                            </button>
                        </div>

                        {/* Schedule Appointment */}
                        <div className="booking-card" onClick={() => isAuthenticated ? navigate('/hospitals') : openAuth('login')}>
                            <div className="booking-card-glow booking-glow-purple"></div>
                            <div className="booking-card-icon-wrap booking-icon-purple">
                                <span>📅</span>
                            </div>
                            <h3>Schedule Appointment</h3>
                            <p>Book consultations with sign-language certified physicians. Video, in-person, or hybrid appointments available.</p>
                            <div className="booking-card-meta">
                                <span className="booking-meta-tag booking-tag-purple">🤟 ASL-Certified Doctors</span>
                            </div>
                            <button className="booking-card-btn booking-btn-purple">
                                {isAuthenticated ? 'Schedule Now' : 'Sign In to Book'} →
                            </button>
                        </div>

                        {/* Request Interpreter */}
                        <div className="booking-card" onClick={() => isAuthenticated ? navigate('/communication') : openAuth('login')}>
                            <div className="booking-card-glow booking-glow-gradient"></div>
                            <div className="booking-card-icon-wrap booking-icon-gradient">
                                <span>🧑‍⚕️</span>
                            </div>
                            <h3>Request Interpreter</h3>
                            <p>Connect with certified medical sign language interpreters for your upcoming appointment or ongoing consultation.</p>
                            <div className="booking-card-meta">
                                <span className="booking-meta-tag booking-tag-gradient">📡 Available 24/7</span>
                            </div>
                            <button className="booking-card-btn booking-btn-gradient">
                                {isAuthenticated ? 'Request Now' : 'Sign In to Book'} →
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            {/* ═══ CAPABILITIES ═══ */}
            <section className="capabilities-section" id="accessibility">
                <div className="capabilities-container">
                    <div className="capabilities-header">
                        <span className="section-label section-label-purple">Platform Capabilities</span>
                        <h2 className="capabilities-title">Enterprise-Grade Accessibility Infrastructure</h2>
                        <p className="capabilities-subtitle">
                            Purpose-built technology stack designed to meet the most rigorous standards
                            in accessible healthcare delivery
                        </p>
                    </div>
                    <div className="capabilities-grid">
                        {capabilities.map((item, index) => (
                            <div
                                key={index}
                                className="capability-card"
                                style={{ animationDelay: `${index * 0.1}s` }}
                            >
                                <span className="capability-icon">{item.icon}</span>
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
                            <h3>Critical Care Access — No Authentication Required</h3>
                            <p>Immediate access to emergency dispatch, ambulance tracking, and nearest facility location services</p>
                        </div>
                    </div>
                    <Link to="/ambulance" className="emergency-cta" id="cta-emergency-access">
                        Access Emergency Services →
                    </Link>
                </div>
            </section>

            {/* ═══ FOOTER ═══ */}
            <footer className="home-footer">
                <div className="footer-container">
                    <div className="footer-top">
                        <div className="footer-brand">
                            <span className="footer-logo">⚕</span>
                            <span className="footer-name">AuraCare</span>
                            <p className="footer-tagline">Next-Generation Accessible Healthcare Technology</p>
                        </div>
                        <div className="footer-links">
                            <div className="footer-col">
                                <h4>Clinical Services</h4>
                                <a href="#services">Sign Language Recognition</a>
                                <a href="#services">Medical Transcription</a>
                                <a href="#services">Document Intelligence</a>
                            </div>
                            <div className="footer-col">
                                <h4>Compliance</h4>
                                <a href="#accessibility">WCAG 2.1 AA</a>
                                <a href="#accessibility">ADA Section 508</a>
                                <a href="#accessibility">HIPAA Standards</a>
                            </div>
                            <div className="footer-col">
                                <h4>Quick Access</h4>
                                <Link to="/ambulance">Emergency Services</Link>
                                <a href="#services">Provider Directory</a>
                                <a href="#hero">Platform Overview</a>
                            </div>
                        </div>
                    </div>
                    <div className="footer-bottom">
                        <p>© 2025 AuraCare Medical Technologies. All rights reserved.</p>
                        <div className="footer-compliance-tags">
                            <span className="footer-tag">♿ WCAG 2.1</span>
                            <span className="footer-tag">🔒 HIPAA</span>
                            <span className="footer-tag">🤟 ASL/ISL</span>
                            <span className="footer-tag">🔬 FDA</span>
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
