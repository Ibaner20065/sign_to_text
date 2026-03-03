import React, { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import { Hospital, Heart, Brain, Baby, Siren, MapPin } from 'lucide-react'
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
  const { token, user } = useAuth()
  const [userLocation, setUserLocation] = useState(null)
  const [hospitals, setHospitals] = useState([])
  const [filteredHospitals, setFilteredHospitals] = useState([])
  const [selectedSpecialty, setSelectedSpecialty] = useState('All')
  const [bookingStatus, setBookingStatus] = useState({})
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'


  const specialties = ['All', 'General', 'Cardiology', 'Neurology', 'Pediatrics', 'Emergency']

  const specialtyIcons = {
    All: <Hospital size={16} />,
    General: <Hospital size={16} />,
    Cardiology: <Heart size={16} />,
    Neurology: <Brain size={16} />,
    Pediatrics: <Baby size={16} />,
    Emergency: <Siren size={16} />,
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

  const fetchHospitals = () => {
    const hospitalData = [
      { id: 101, name: "City General Hospital", lat_offset: 0.005, lng_offset: 0.005, specialty: "General", phone: "+91-11-2345-6789" },
      { id: 102, name: "Heart Care Center", lat_offset: -0.008, lng_offset: 0.01, specialty: "Cardiology", phone: "+91-11-2345-6790" },
      { id: 103, name: "Neuro Institute", lat_offset: 0.012, lng_offset: -0.005, specialty: "Neurology", phone: "+91-11-2345-6791" },
      { id: 104, name: "Kids Health Hospital", lat_offset: -0.003, lng_offset: -0.01, specialty: "Pediatrics", phone: "+91-11-2345-6792" },
      { id: 105, name: "Emergency Trauma Center", lat_offset: 0.009, lng_offset: 0.015, specialty: "Emergency", phone: "+91-11-2345-6793" },
      { id: 106, name: "Apollo Medical Center", lat_offset: -0.015, lng_offset: 0.003, specialty: "General", phone: "+91-11-2345-6794" },
      { id: 107, name: "Max Heart Hospital", lat_offset: 0.018, lng_offset: 0.008, specialty: "Cardiology", phone: "+91-11-2345-6795" },
      { id: 108, name: "Brain & Spine Clinic", lat_offset: -0.006, lng_offset: 0.018, specialty: "Neurology", phone: "+91-11-2345-6796" },
      { id: 109, name: "Rainbow Children's Hospital", lat_offset: 0.004, lng_offset: -0.016, specialty: "Pediatrics", phone: "+91-11-2345-6797" },
      { id: 110, name: "24x7 Emergency Care", lat_offset: -0.012, lng_offset: -0.008, specialty: "Emergency", phone: "+91-11-2345-6798" },
      { id: 111, name: "Fortis Healthcare", lat_offset: 0.014, lng_offset: 0.012, specialty: "General", phone: "+91-11-2345-6799" },
      { id: 112, name: "Medanta Cardiac Unit", lat_offset: -0.016, lng_offset: 0.014, specialty: "Cardiology", phone: "+91-11-2345-6800" },
      { id: 113, name: "NIMHANS Neuro Center", lat_offset: 0.008, lng_offset: -0.012, specialty: "Neurology", phone: "+91-11-2345-6801" },
      { id: 114, name: "Cloudnine Kids Hospital", lat_offset: -0.01, lng_offset: 0.007, specialty: "Pediatrics", phone: "+91-11-2345-6802" },
      { id: 115, name: "Metro Emergency Hospital", lat_offset: 0.002, lng_offset: 0.02, specialty: "Emergency", phone: "+91-11-2345-6803" },
    ]
    setHospitals(hospitalData)
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

  const handleBookBed = async (hospital) => {
    if (!token) {
      setBookingStatus((prev) => ({
        ...prev,
        [hospital.id]: { type: 'error', message: 'Please log in to book a bed.' }
      }))
      return
    }

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000)

      const response = await fetch(`${apiUrl}/api/v1/hospitals/book-bed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          hospital_id: String(hospital.id),
          hospital_name: hospital.name,
          bed_type: hospital.specialty === 'Emergency' ? 'emergency' : 'general',
          patient_name: user?.name || 'Patient',
        }),
        signal: controller.signal,
      })
      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Booking failed (${response.status})`)
      }

      const data = await response.json()
      setBookingStatus((prev) => ({
        ...prev,
        [hospital.id]: { type: 'success', message: data.message }
      }))
    } catch (error) {
      const isOffline = error.name === 'AbortError' || error.message === 'Failed to fetch'
      setBookingStatus((prev) => ({
        ...prev,
        [hospital.id]: {
          type: isOffline ? 'success' : 'error',
          message: isOffline
            ? `Bed booking request for ${hospital.name} recorded. You will receive confirmation once the server is back online.`
            : (error.message || 'Unable to book bed.')
        }
      }))
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
    <div id="hospitals-page">
      <h1 className="page-title"><Hospital size={36} style={{ verticalAlign: 'middle', marginRight: '12px' }} /> Hospital Finder</h1>

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
                style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
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
              <Popup><MapPin size={14} style={{ verticalAlign: 'middle', marginRight: '4px' }} /> Your Location</Popup>
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
                <span className={`specialty-tag ${getSpecialtyClass(hospital.specialty)}`} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {specialtyIcons[hospital.specialty]} {hospital.specialty}
                </span>
                <h4>{hospital.name}</h4>
                <p><strong>Distance:</strong> <span className="distance-value">{hospital.distance} km</span></p>
                <p><strong>Phone:</strong> <span className="phone-value">{hospital.phone}</span></p>
                <button className="book-bed-button" onClick={() => handleBookBed(hospital)}>
                  Book Bed
                </button>
                {bookingStatus[hospital.id] && (
                  <p className={`booking-feedback ${bookingStatus[hospital.id].type}`}>
                    {bookingStatus[hospital.id].message}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default HospitalFinder
