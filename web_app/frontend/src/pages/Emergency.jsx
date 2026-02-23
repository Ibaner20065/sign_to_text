import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import AmbulanceTracker from './AmbulanceTracker'
import HospitalFinder from './HospitalFinder'
import './Emergency.css'

const Emergency = () => {
    const [activeTab, setActiveTab] = useState('hospitals')

    return (
        <div className="emergency-page">
            <div className="emergency-header">
                <div className="emergency-banner" id="emergency-banner">
                    <span className="emergency-pulse">🚨</span>
                    <div>
                        <h1>Emergency Access</h1>
                        <p>Find nearby hospitals & track ambulances — no login required</p>
                    </div>
                </div>
                <Link to="/login" className="back-to-login" id="back-to-login">
                    ← Back to Login
                </Link>
            </div>

            <div className="emergency-tabs" id="emergency-tabs">
                <button
                    className={`emergency-tab ${activeTab === 'hospitals' ? 'active' : ''}`}
                    onClick={() => setActiveTab('hospitals')}
                    id="tab-hospitals"
                >
                    🏥 Find Hospitals
                </button>
                <button
                    className={`emergency-tab ${activeTab === 'ambulance' ? 'active' : ''}`}
                    onClick={() => setActiveTab('ambulance')}
                    id="tab-ambulance"
                >
                    🚑 Track Ambulance
                </button>
            </div>

            <div className="emergency-content">
                {activeTab === 'hospitals' ? <HospitalFinder /> : <AmbulanceTracker />}
            </div>
        </div>
    )
}

export default Emergency
