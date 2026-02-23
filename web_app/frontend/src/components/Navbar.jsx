import React, { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import {
  Hand,
  Truck,
  FileText,
  Hospital as HospitalIcon,
  HeartPulse,
  User as UserIcon,
  LogOut,
  Menu,
  X,
  Activity
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import AuthModal from './AuthModal'
import ColorBlindFilter from './ColorBlindFilter'
import './Navbar.css'

const Navbar = ({ variant = 'auto' }) => {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, isAuthenticated, logout } = useAuth()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [authMode, setAuthMode] = useState('login')

  const isHome = location.pathname === '/'
  const showDashboard = variant === 'dashboard' || (variant === 'auto' && isAuthenticated && !isHome)

  const openAuth = (mode) => {
    setAuthMode(mode)
    setShowAuthModal(true)
    setMobileOpen(false)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/')
    setMobileOpen(false)
  }

  // Dashboard nav items
  const dashboardItems = [
    { path: '/communication', label: 'Communication', icon: <Hand size={20} /> },
    { path: '/ambulance', label: 'Ambulance', icon: <Truck size={20} /> },
    { path: '/scanner', label: 'Scanner', icon: <FileText size={20} /> },
    { path: '/hospitals', label: 'Hospitals', icon: <HospitalIcon size={20} /> },
    { path: '/health', label: 'Health', icon: <HeartPulse size={20} /> },
    { path: '/profile', label: 'Profile', icon: <UserIcon size={20} /> },
  ]

  // Public nav items
  const publicItems = [
    { href: '#services', label: 'Services' },
    { href: '#book', label: 'Book Services' },
    { href: '#accessibility', label: 'Accessibility' },
    { path: '/emergency', label: 'Emergency' },
  ]

  return (
    <>
      <nav className={`navbar ${showDashboard ? 'navbar-dashboard' : 'navbar-public'}`} id="main-navbar">
        <div className="navbar-container">
          {/* ─── Brand ─── */}
          <Link to="/" className="navbar-brand" id="navbar-brand" onClick={() => setMobileOpen(false)}>
            <div className="brand-logo-container">
              <Activity className="brand-activity-icon" size={28} />
            </div>
            <span className="brand-text">AuraCare</span>
          </Link>

          {/* ─── Mobile Toggle ─── */}
          <button
            className="navbar-toggle"
            id="navbar-toggle"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle navigation"
          >
            {mobileOpen ? <X size={24} /> : <Menu size={24} />}
          </button>

          {/* ─── Navigation Links ─── */}
          <div className={`navbar-links ${mobileOpen ? 'mobile-open' : ''}`} id="navbar-links">
            {showDashboard ? (
              // Dashboard mode
              dashboardItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`navbar-link ${location.pathname === item.path ? 'active' : ''}`}
                  id={`nav-${item.path.slice(1)}`}
                  onClick={() => setMobileOpen(false)}
                >
                  <span className="navbar-icon">{item.icon}</span>
                  <span className="navbar-label">{item.label}</span>
                  {location.pathname === item.path && <span className="active-indicator"></span>}
                </Link>
              ))
            ) : (
              // Public mode
              publicItems.map((item, i) =>
                item.path ? (
                  <Link
                    key={i}
                    to={item.path}
                    className="navbar-link"
                    onClick={() => setMobileOpen(false)}
                  >
                    {item.label}
                  </Link>
                ) : (
                  <a
                    key={i}
                    href={item.href}
                    className="navbar-link"
                    onClick={() => setMobileOpen(false)}
                  >
                    {item.label}
                  </a>
                )
              )
            )}
          </div>

          {/* ─── Auth / User Area ─── */}
          <div className="navbar-auth" id="navbar-auth">
            <ColorBlindFilter />
            {isAuthenticated ? (
              <>
                {!showDashboard && (
                  <Link to="/communication" className="navbar-dashboard-link" onClick={() => setMobileOpen(false)}>
                    Dashboard
                  </Link>
                )}
                <div className="navbar-user" id="navbar-user">
                  <div className="user-avatar">
                    {user?.name?.charAt(0)?.toUpperCase() || '?'}
                  </div>
                  <span className="user-name">{user?.name || 'User'}</span>
                </div>
                <button className="navbar-logout" onClick={handleLogout} aria-label="Sign out">
                  <LogOut size={18} />
                </button>
              </>
            ) : (
              <>
                <button className="navbar-auth-btn navbar-login" onClick={() => openAuth('login')} id="nav-login">
                  Log In
                </button>
                <button className="navbar-auth-btn navbar-signup" onClick={() => openAuth('register')} id="nav-signup">
                  Sign Up
                </button>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Auth Modal */}
      {showAuthModal && (
        <AuthModal
          mode={authMode}
          onClose={() => setShowAuthModal(false)}
          onSwitchMode={(mode) => setAuthMode(mode)}
        />
      )}
    </>
  )
}

export default Navbar
