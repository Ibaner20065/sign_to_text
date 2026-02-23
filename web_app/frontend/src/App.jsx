import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Login from './pages/Login'
import Register from './pages/Register'
import Communication from './pages/Communication'
import AmbulanceTracker from './pages/AmbulanceTracker'
import DocumentScanner from './pages/DocumentScanner'
import HospitalFinder from './pages/HospitalFinder'
import Profile from './pages/Profile'
import Emergency from './pages/Emergency'
import AIChatbot from './components/AIChatbot'
import Health from './pages/Health'
import './App.css'

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="app">
          <Routes>
            {/* Public Routes */}
            <Route
              path="/"
              element={
                <>
                  <Navbar variant="public" />
                  <Home />
                </>
              }
            />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/emergency" element={<Emergency />} />

            {/* Protected Dashboard Routes */}
            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <div>
                    <Navbar variant="dashboard" />
                    <div className="content">
                      <Routes>
                        <Route path="/communication" element={<Communication />} />
                        <Route path="/ambulance" element={<AmbulanceTracker />} />
                        <Route path="/scanner" element={<DocumentScanner />} />
                        <Route path="/hospitals" element={<HospitalFinder />} />
                        <Route path="/health" element={<Health />} />
                        <Route path="/profile" element={<Profile />} />
                      </Routes>
                    </div>
                  </div>
                </ProtectedRoute>
              }
            />
          </Routes>
          <AIChatbot />
        </div>
      </Router>
    </AuthProvider>
  )
}

export default App
