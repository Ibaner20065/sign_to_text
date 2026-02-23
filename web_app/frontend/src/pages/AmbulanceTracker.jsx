import React, { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'

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
  const [userLocation, setUserLocation] = useState(null)
  const [ambulances, setAmbulances] = useState([])
  const [showSOSModal, setShowSOSModal] = useState(false)
  const [sosSent, setSosSent] = useState(false)

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
    }
  }

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
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p className="loading-text">Getting your location...</p>
      </div>
    )
  }

  return (
    <div id="ambulance-page">
      <h1 className="page-title">🚑 Ambulance Tracker</h1>

      <div className="ambulance-container">
        {/* Ambulance Info Cards */}
        <div className="ambulance-info-grid" id="ambulance-info-grid">
          {ambulances.map((amb) => (
            <div key={amb.id} className="ambulance-info-card">
              <div className={`amb-icon-wrap ${getAmbTypeClass(amb.type)}`}>
                🚑
              </div>
              <div className="amb-details">
                <h4>Ambulance #{amb.id} — {amb.type}</h4>
                <p>
                  Status: <span className={`badge ${amb.status === 'Moving' ? 'badge-success' : 'badge-warning'}`}>
                    {amb.status}
                  </span>
                </p>
                <p className="amb-eta">ETA: {amb.eta}</p>
              </div>
            </div>
          ))}
        </div>

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
            {ambulances.map((amb) => (
              <Marker key={amb.id} position={[amb.lat, amb.lng]} icon={ambulanceIcon}>
                <Popup>
                  <div className="ambulance-popup">
                    <h3>🚑 Ambulance #{amb.id}</h3>
                    <p><strong>Status:</strong> {amb.status}</p>
                    <p><strong>Type:</strong> {amb.type}</p>
                    <p><strong>ETA:</strong> {amb.eta}</p>
                  </div>
                </Popup>
              </Marker>
            ))}
            <Marker position={userLocation} icon={userIcon}>
              <Popup>📍 Your Location</Popup>
            </Marker>
          </MapContainer>
        </div>

        {/* SOS Section */}
        <div className="sos-section" id="sos-section">
          <div className="sos-button-wrapper">
            <button className="sos-button" onClick={handleSOS} id="sos-button">
              🚨 EMERGENCY SOS
            </button>
          </div>
          {sosSent && (
            <div className="sos-message" id="sos-confirmation">
              ✅ Emergency alert sent! Help is on the way. Nearest ambulance arriving in ~5 minutes.
            </div>
          )}
        </div>
      </div>

      {/* SOS Confirmation Modal */}
      {showSOSModal && (
        <div className="modal-overlay" onClick={() => setShowSOSModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} id="sos-modal">
            <h2>⚠️ Confirm Emergency SOS</h2>
            <p>This will send an emergency alert to all nearby ambulance services. Are you sure you want to proceed?</p>
            <div className="modal-buttons">
              <button className="button button-danger" onClick={confirmSOS} id="sos-confirm">
                🚨 Yes, Send SOS
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
