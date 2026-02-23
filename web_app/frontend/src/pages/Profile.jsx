import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { User, Camera, Mail, Settings, Volume2, Lock, Key, LogOut, CheckCircle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import './Profile.css'

const Profile = () => {
  const { user, logout, changePassword } = useAuth()
  const navigate = useNavigate()
  const [settings, setSettings] = useState({
    fontSize: 'normal',
    highContrast: false,
    ttsSpeed: 1.0,
  })
  const [profilePic, setProfilePic] = useState(null)

  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [message, setMessage] = useState('')
  const [messageType, setMessageType] = useState('info')

  useEffect(() => {
    const savedSettings = localStorage.getItem('userSettings')
    if (savedSettings) {
      const parsed = JSON.parse(savedSettings)
      setSettings(parsed)
      applySettings(parsed)
    }

    const savedPic = localStorage.getItem('profilePic')
    if (savedPic) {
      setProfilePic(savedPic)
    }
  }, [])

  useEffect(() => {
    applySettings(settings)
    localStorage.setItem('userSettings', JSON.stringify(settings))
  }, [settings])

  const applySettings = (currentSettings) => {
    const root = document.documentElement
    root.classList.remove('font-normal', 'font-large', 'font-extra-large', 'high-contrast')
    root.classList.add(`font-${currentSettings.fontSize}`)
    if (currentSettings.highContrast) {
      root.classList.add('high-contrast')
    }
  }

  const handleFontSizeChange = (size) => {
    setSettings({ ...settings, fontSize: size })
  }

  const handleHighContrastToggle = () => {
    setSettings({ ...settings, highContrast: !settings.highContrast })
  }

  const handleTtsSpeedChange = (speed) => {
    setSettings({ ...settings, ttsSpeed: parseFloat(speed) })
  }

  const handleProfilePicChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setProfilePic(reader.result)
        localStorage.setItem('profilePic', reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePasswordChange = async (e) => {
    e.preventDefault()
    setMessage('')

    if (newPassword !== confirmPassword) {
      setMessage('New passwords do not match')
      setMessageType('error')
      return
    }

    if (newPassword.length < 6) {
      setMessage('Password must be at least 6 characters')
      setMessageType('error')
      return
    }

    const result = await changePassword('', newPassword)
    if (result.success) {
      setMessage('Password changed successfully! 🎉')
      setMessageType('success')

      setNewPassword('')
      setConfirmPassword('')
    } else {
      setMessage(result.error || 'Password change failed')
      setMessageType('error')
    }
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div id="profile-page">
      <h1 className="page-title"><User size={36} style={{ verticalAlign: 'middle', marginRight: '12px' }} /> Profile & Settings</h1>

      <div className="profile-container">
        {/* Profile Info */}
        <div className="card">
          <div className="profile-section" id="profile-info">
            <div className="profile-pic-section">
              {profilePic ? (
                <img src={profilePic} alt="Profile" className="profile-pic" />
              ) : (
                <div className="profile-pic-placeholder">
                  {user?.name?.charAt(0)?.toUpperCase() || <User />}
                </div>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleProfilePicChange}
                className="file-input"
                id="profile-pic-input"
              />
              <label htmlFor="profile-pic-input" className="profile-pic-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center' }}>
                <Camera size={18} /> Change Photo
              </label>
            </div>
            <div className="profile-info">
              <p className="profile-name">{user?.name || 'User'}</p>
              <p className="profile-email" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Mail size={16} /> {user?.email || 'user@example.com'}
              </p>
            </div>
          </div>
        </div>

        {/* Accessibility Settings */}
        <div className="card">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><Settings size={22} /> Accessibility Preferences</h2>
          <div className="settings-section" id="accessibility-settings">
            {/* Font Size */}
            <div className="setting-item">
              <div className="setting-header">Font Size</div>
              <p className="text-muted" style={{ fontSize: '0.85rem', marginBottom: 'var(--space-3)' }}>
                Adjust the text size across the entire platform
              </p>
              <div className="font-size-buttons" id="font-size-buttons">
                {[
                  { value: 'normal', label: 'Normal', sample: 'Aa' },
                  { value: 'large', label: 'Large', sample: 'Aa' },
                  { value: 'extra-large', label: 'Extra Large', sample: 'Aa' },
                ].map((size) => (
                  <button
                    key={size.value}
                    className={`size-button ${settings.fontSize === size.value ? 'active' : ''}`}
                    onClick={() => handleFontSizeChange(size.value)}
                    id={`font-${size.value}`}
                  >
                    {size.label}
                  </button>
                ))}
              </div>
            </div>

            {/* High Contrast */}
            <div className="setting-item">
              <div className="toggle-wrapper" onClick={handleHighContrastToggle} id="high-contrast-toggle">
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={settings.highContrast}
                    onChange={handleHighContrastToggle}
                  />
                  <span className="toggle-slider"></span>
                </label>
                <div>
                  <div className="toggle-label">High Contrast Mode</div>
                  <div className="toggle-description">Increase contrast for better visibility</div>
                </div>
              </div>
            </div>

            {/* TTS Speed */}
            <div className="setting-item">
              <div className="setting-header">Text-to-Speech Speed</div>
              <div className="slider-wrapper">
                <div className="slider-value" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Volume2 size={18} /> {settings.ttsSpeed.toFixed(1)}x</div>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={settings.ttsSpeed}
                  onChange={(e) => handleTtsSpeedChange(e.target.value)}
                  className="slider"
                  id="tts-speed-slider"
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 'var(--space-2)' }}>
                  <span className="text-muted" style={{ fontSize: '0.75rem' }}>Slow (0.5x)</span>
                  <span className="text-muted" style={{ fontSize: '0.75rem' }}>Fast (2.0x)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Change Password */}
        <div className="card">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><Lock size={22} /> Change Password</h2>
          <form onSubmit={handlePasswordChange} className="password-form" id="password-form">
            <div className="form-group">
              <label className="label" htmlFor="new-password">New Password</label>
              <input
                id="new-password"
                type="password"
                className="input"
                placeholder="Min. 6 characters"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label className="label" htmlFor="confirm-password">Confirm New Password</label>
              <input
                id="confirm-password"
                type="password"
                className="input"
                placeholder="Re-enter new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
            {message && (
              <div className={`profile-message ${messageType === 'success' ? 'success-message' : messageType === 'error' ? 'error-message' : 'info-message'}`} id="password-message">
                {message}
              </div>
            )}
            <button type="submit" className="button button-primary" id="change-password-btn" style={{ display: 'flex', alignItems: 'center', gap: '10px', justifyContent: 'center' }}>
              <Key size={18} /> Update Password
            </button>
          </form>
        </div>

        {/* Logout */}
        <div className="card">
          <button className="button button-danger logout-button" onClick={handleLogout} id="logout-button" style={{ display: 'flex', alignItems: 'center', gap: '10px', justifyContent: 'center' }}>
            <LogOut size={18} /> Logout
          </button>
        </div>
      </div>
    </div>
  )
}

export default Profile
