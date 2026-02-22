import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Navbar.css'

const Navbar = () => {
  const location = useLocation()
  const { user } = useAuth()
  const [mobileOpen, setMobileOpen] = useState(false)

  const navItems = [
    { path: '/communication', label: 'Communication', icon: '🤟' },
    { path: '/ambulance', label: 'Ambulance', icon: '🚑' },
    { path: '/scanner', label: 'Scanner', icon: '📄' },
    { path: '/hospitals', label: 'Hospitals', icon: '🏥' },
    { path: '/profile', label: 'Profile', icon: '👤' },
  ]

  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-container">
        <Link to="/communication" className="navbar-brand" id="navbar-brand">
          <span className="brand-icon">💙</span>
          <span className="brand-text">AuraCare</span>
        </Link>

        <button
          className="navbar-toggle"
          id="navbar-toggle"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation"
        >
          <span className={`hamburger ${mobileOpen ? 'open' : ''}`}>
            <span></span>
            <span></span>
            <span></span>
          </span>
        </button>

        <div className={`navbar-links ${mobileOpen ? 'mobile-open' : ''}`} id="navbar-links">
          {navItems.map((item) => (
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
          ))}
        </div>

        <div className="navbar-user" id="navbar-user">
          <div className="user-avatar">
            {user?.name?.charAt(0)?.toUpperCase() || '?'}
          </div>
          <span className="user-name">{user?.name || 'User'}</span>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
