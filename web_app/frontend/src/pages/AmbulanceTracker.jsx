import React, { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import { Truck, Siren, CheckCircle, AlertTriangle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

import './AmbulanceTracker.css'

// Fix for default marker icon
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const ambulanceIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
})

const userIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
})

function MapUpdater({ center }) {
  const map = useMap()
  useEffect(() => {
    map.setView(center, map.getZoom())
  }, [center, map])
  return null
}

const AmbulanceTracker = () => {
  const { token } = useAuth()
  const [userLocation, setUserLocation] = useState(null)
  const [ambulances, setAmbulances] = useState([])
  const [showSOSModal, setShowSOSModal] = useState(false)
  const [sosSent, setSosSent] = useState(false)
  const [bookingStage, setBookingStage] = useState('idle') // idle, assigning, enroute, arrived
  const [bookedAmbId, setBookedAmbId] = useState(null)
  const [bookingId, setBookingId] = useState(null)
  const [bookingMessage, setBookingMessage] = useState('')
  const [bookingError, setBookingError] = useState('')
  const [simLocation, setSimLocation] = useState(null)
  const [simETA, setSimETA] = useState(null)
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  const intervalRef = useRef(null)

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation([position.coords.latitude, position.coords.longitude])
        },
        (error) => {
          console.error('Geolocation error:', error)
          setUserLocation([28.6139, 77.209])
        }
      )
    } else {
      setUserLocation([28.6139, 77.209])
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (userLocation) {
      fetchAmbulances()
      intervalRef.current = setInterval(() => {
        fetchAmbulances()
      }, 5000)
    }
  }, [userLocation])

  const fetchAmbulances = () => {
    const ambulanceData = [
      {
        id: 1,
        lat_offset: 0.01 + (Math.random() - 0.5) * 0.004,
        lng_offset: 0.01 + (Math.random() - 0.5) * 0.004,
        status: "Moving",
        eta: "5 mins",
        type: "ALS"
      },
      {
        id: 2,
        lat_offset: -0.01 + (Math.random() - 0.5) * 0.004,
        lng_offset: 0.02 + (Math.random() - 0.5) * 0.004,
        status: "Stationary",
        eta: "12 mins",
        type: "BLS"
      },
      {
        id: 3,
        lat_offset: 0.02 + (Math.random() - 0.5) * 0.004,
        lng_offset: -0.01 + (Math.random() - 0.5) * 0.004,
        status: "Moving",
        eta: "8 mins",
        type: "Neonatal"
      }
    ]

    if (userLocation) {
      const updatedAmbulances = ambulanceData.map((amb) => ({
        ...amb,
        lat: userLocation[0] + amb.lat_offset,
        lng: userLocation[1] + amb.lng_offset,
      }))
      setAmbulances(updatedAmbulances)

      // Initialize simulation if first run
      if (bookingStage === 'idle' && !simLocation) {
        // Just for visual baseline
      }
    }
  }

  const handleBook = async (amb) => {
    setBookingError('')
    if (!token) {
      setBookingError('Please log in to book an ambulance.')
      return
    }

    try {
      const response = await fetch(`${apiUrl}/api/v1/ambulance/book`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          latitude: userLocation[0],
          longitude: userLocation[1],
          emergency_note: 'Emergency booking from tracker',
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Booking failed (${response.status})`)
      }

      const data = await response.json()
      setBookedAmbId(amb.id)
      setBookingId(data.booking_id)
      setBookingStage(data.status || 'assigning')
      setBookingMessage(data.message || 'Ambulance dispatch initiated.')
      setSimLocation([amb.lat, amb.lng])
      setSimETA(data.eta_minutes || parseInt(amb.eta, 10) || 10)
    } catch (error) {
      setBookingError(error.message || 'Unable to book ambulance right now.')
    }
  }

  useEffect(() => {
    if (!bookingId || !token) {
      return undefined
    }

    const poller = setInterval(async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/ambulance/booking/${bookingId}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        })
        if (!response.ok) {
          return
        }

        const data = await response.json()
        setBookingStage(data.status)
        setSimETA(data.eta_minutes)
        setBookingMessage(data.message)

        if (data.status === 'arrived') {
          setSimLocation(userLocation)
          clearInterval(poller)
        }
      } catch (_error) {
      }
    }, 5000)

    return () => clearInterval(poller)
  }, [bookingId, token, apiUrl, userLocation])

  // Simulation Animation Logic
  useEffect(() => {
    if (bookingStage === 'enroute' && simLocation && userLocation) {
      const timer = setInterval(() => {
        setSimLocation(current => {
          const latDiff = userLocation[0] - current[0]
          const lngDiff = userLocation[1] - current[1]

          // Move 10% of the way each tick
          const nextLat = current[0] + latDiff * 0.1
          const nextLng = current[1] + lngDiff * 0.1

          // Check if arrived
          if (Math.abs(latDiff) < 0.0005 && Math.abs(lngDiff) < 0.0005) {
            setBookingStage('arrived')
            clearInterval(timer)
            return userLocation
          }

          return [nextLat, nextLng]
        })

        setSimETA(prev => (prev > 1 ? prev - 0.2 : 1))
      }, 1000)

      return () => clearInterval(timer)
    }
  }, [bookingStage, userLocation])

  const handleSOS = () => {
    setShowSOSModal(true)
  }

  const confirmSOS = () => {
    setShowSOSModal(false)
    setSosSent(true)
    setTimeout(() => {
      setSosSent(false)
    }, 5000)
  }

  const getAmbTypeClass = (type) => {
    switch (type) {
      case 'ALS': return 'als'
      case 'BLS': return 'bls'
      case 'Neonatal': return 'neonatal'
      default: return 'als'
    }
  }

  if (!userLocation) {
    return (
      <div className="loading-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg-primary)' }}>
        <div className="loading-spinner" style={{ width: '40px', height: '40px', border: '4px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--accent-400)', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
        <p className="loading-text" style={{ marginTop: '20px', color: 'var(--text-secondary)' }}>Initializing Neural GPS...</p>
      </div>
    )
  }

  return (
    <div id="ambulance-page">
      <h1 className="page-title"><Truck size={36} style={{ verticalAlign: 'middle', marginRight: '12px' }} /> Ambulance Tracker</h1>

      <div className="ambulance-container">
        {/* Ambulance Info Cards */}
        <div className="ambulance-info-grid" id="ambulance-info-grid">
          {ambulances.map((amb) => (
            <div key={amb.id} className={`ambulance-info-card ${bookedAmbId === amb.id ? 'active-booking' : ''}`}>
              <div className={`amb-icon-wrap ${getAmbTypeClass(amb.type)}`}>
                <Truck size={24} />
              </div>
              <div className="amb-details">
                <h4>Ambulance #{amb.id} — {amb.type}</h4>
                <p>
                  Status: <span className={`badge ${amb.status === 'Moving' ? 'badge-success' : 'badge-warning'}`}>
                    {bookedAmbId === amb.id && bookingStage !== 'idle' ? 'Dispatched' : amb.status}
                  </span>
                </p>
                <p className="amb-eta">
                  ETA: {bookedAmbId === amb.id && bookingStage === 'enroute' ? `${Math.ceil(simETA)} mins` : amb.eta}
                </p>
              </div>
              {bookingStage === 'idle' && (
                <button className="book-mini-button" onClick={() => handleBook(amb)}>Book</button>
              )}
            </div>
          ))}
        </div>

        {bookingMessage && (
          <div className="booking-status-banner" id="booking-status-banner">
            {bookingMessage}
          </div>
        )}

        {bookingError && (
          <div className="booking-error-banner" id="booking-error-banner">
            {bookingError}
          </div>
        )}

        {/* Map */}
        <div className="map-wrapper" id="ambulance-map">
          <MapContainer
            center={userLocation}
            zoom={13}
            style={{ height: '450px', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <MapUpdater center={userLocation} />
            {ambulances.map((amb) => {
              // Hide the original marker if it's being "simulated" separately to avoid double markers
              if (bookedAmbId === amb.id && bookingStage !== 'idle') return null;

              return (
                <Marker key={amb.id} position={[amb.lat, amb.lng]} icon={ambulanceIcon}>
                  <Popup>
                    <div className="ambulance-popup">
                      <h3><Truck size={20} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Ambulance #{amb.id}</h3>
                      <p><strong>Status:</strong> {amb.status}</p>
                      <p><strong>Type:</strong> {amb.type}</p>
                      <p><strong>ETA:</strong> {amb.eta}</p>
                      {bookingStage === 'idle' && (
                        <button className="booking-cta-btn" onClick={() => handleBook(amb)}>Assign to Me</button>
                      )}
                    </div>
                  </Popup>
                </Marker>
              )
            })}

            {/* Simulated Active Ambulance Marker */}
            {bookingStage !== 'idle' && simLocation && (
              <Marker position={simLocation} icon={ambulanceIcon}>
                <Popup>
                  <div className="ambulance-popup active-dispatch">
                    <h3>🚑 EMERGENCY DISPATCH</h3>
                    <p><strong>Tracking:</strong> Ambulance #{bookedAmbId}</p>
                    <p><strong>Status:</strong> {bookingStage === 'enroute' ? 'En Route' : 'Arrived'}</p>
                    <p><strong>ETA:</strong> {bookingStage === 'arrived' ? 'READY' : `${Math.ceil(simETA)} mins`}</p>
                  </div>
                </Popup>
              </Marker>
            )}
            <Marker position={userLocation} icon={userIcon}>
              <Popup>📍 Your Location</Popup>
            </Marker>
          </MapContainer>
        </div>

        {/* SOS Section */}
        <div className="sos-section" id="sos-section">
          <div className="sos-button-wrapper">
            <button className="sos-button" onClick={handleSOS} id="sos-button" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
              <Siren size={32} /> EMERGENCY SOS
            </button>
          </div>
          {sosSent && (
            <div className="sos-message" id="sos-confirmation" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <CheckCircle size={20} /> Emergency alert sent! Help is on the way. Nearest ambulance arriving in ~5 minutes.
            </div>
          )}
        </div>
      </div>

      {/* SOS Confirmation Modal */}
      {showSOSModal && (
        <div className="modal-overlay" onClick={() => setShowSOSModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} id="sos-modal">
            <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}><AlertTriangle size={32} color="#f59e0b" /> Confirm Emergency SOS</h2>
            <p>This will send an emergency alert to all nearby ambulance services. Are you sure you want to proceed?</p>
            <div className="modal-buttons">
              <button className="button button-danger" onClick={confirmSOS} id="sos-confirm" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Siren size={18} /> Yes, Send SOS
              </button>
              <button className="button button-secondary" onClick={() => setShowSOSModal(false)} id="sos-cancel">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AmbulanceTracker
