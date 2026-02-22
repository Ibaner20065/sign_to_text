import React, { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import { useAuth } from '../context/AuthContext'
import './HospitalFinder.css'

// Fix for default marker icon
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const hospitalIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
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

const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371
  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
    Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLon / 2) *
    Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return (R * c).toFixed(2)
}

const getSpecialtyClass = (specialty) => {
  return specialty.toLowerCase()
}

const HospitalFinder = () => {
  const [userLocation, setUserLocation] = useState(null)
  const [hospitals, setHospitals] = useState([])
  const [filteredHospitals, setFilteredHospitals] = useState([])
  const [selectedSpecialty, setSelectedSpecialty] = useState('All')
  const { token } = useAuth()

  const specialties = ['All', 'General', 'Cardiology', 'Neurology', 'Pediatrics', 'Emergency']

  const specialtyIcons = {
    All: '🏥',
    General: '🏥',
    Cardiology: '❤️',
    Neurology: '🧠',
    Pediatrics: '👶',
    Emergency: '🚨',
  }

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

    fetchHospitals()
  }, [])

  useEffect(() => {
    if (userLocation && hospitals.length > 0) {
      filterHospitals()
    }
  }, [selectedSpecialty, hospitals, userLocation])

  const fetchHospitals = async () => {
    try {
      const response = await fetch('/api/hospitals', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
      const data = await response.json()
      setHospitals(data.hospitals)
    } catch (error) {
      console.error('Error fetching hospitals:', error)
    }
  }

  const filterHospitals = () => {
    let filtered = hospitals

    if (selectedSpecialty !== 'All') {
      filtered = hospitals.filter((h) => h.specialty === selectedSpecialty)
    }

    if (userLocation) {
      filtered = filtered.map((hospital) => {
        const lat = userLocation[0] + hospital.lat_offset
        const lng = userLocation[1] + hospital.lng_offset
        const distance = calculateDistance(userLocation[0], userLocation[1], lat, lng)
        return { ...hospital, lat, lng, distance: parseFloat(distance) }
      })

      filtered.sort((a, b) => a.distance - b.distance)
    }

    setFilteredHospitals(filtered)
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
    <div id="hospitals-page">
      <h1 className="page-title">🏥 Hospital Finder</h1>

      <div className="hospital-container">
        <div className="filter-section" id="filter-section">
          <label className="label">Filter by Specialty</label>
          <div className="specialty-buttons">
            {specialties.map((specialty) => (
              <button
                key={specialty}
                className={`specialty-button ${selectedSpecialty === specialty ? 'active' : ''}`}
                onClick={() => setSelectedSpecialty(specialty)}
                id={`filter-${specialty.toLowerCase()}`}
              >
                {specialtyIcons[specialty]} {specialty}
              </button>
            ))}
          </div>
        </div>

        <div className="map-wrapper" id="hospital-map">
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
            {filteredHospitals.map((hospital) => (
              <Marker key={hospital.id} position={[hospital.lat, hospital.lng]} icon={hospitalIcon}>
                <Popup>
                  <div className="hospital-popup">
                    <h3>{hospital.name}</h3>
                    <p><strong>Specialty:</strong> {hospital.specialty}</p>
                    <p><strong>Distance:</strong> {hospital.distance} km</p>
                    <p><strong>Phone:</strong> {hospital.phone}</p>
                  </div>
                </Popup>
              </Marker>
            ))}
            <Marker position={userLocation} icon={userIcon}>
              <Popup>📍 Your Location</Popup>
            </Marker>
          </MapContainer>
        </div>

        <div className="hospital-list" id="hospital-list">
          <h3>
            Nearby Hospitals
            <span className="hospital-count">{filteredHospitals.length} found</span>
          </h3>
          <div className="list-container">
            {filteredHospitals.map((hospital) => (
              <div key={hospital.id} className="hospital-card">
                <span className={`specialty-tag ${getSpecialtyClass(hospital.specialty)}`}>
                  {specialtyIcons[hospital.specialty]} {hospital.specialty}
                </span>
                <h4>{hospital.name}</h4>
                <p><strong>Distance:</strong> <span className="distance-value">{hospital.distance} km</span></p>
                <p><strong>Phone:</strong> <span className="phone-value">{hospital.phone}</span></p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default HospitalFinder
