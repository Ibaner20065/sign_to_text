import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { Siren, Hospital, Truck, ArrowLeft } from 'lucide-react'
import AmbulanceTracker from './AmbulanceTracker'
import HospitalFinder from './HospitalFinder'
import './Emergency.css'

const Emergency = () => {
    const [activeTab, setActiveTab] = useState('hospitals')

    return (
        <div className="emergency-page">
            <div className="emergency-header">
                <div className="emergency-banner" id="emergency-banner" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div className="emergency-pulse"><Siren size={40} /></div>
                    <div>
                        <h1>Emergency Access</h1>
                        <p>Find nearby hospitals & track ambulances — no login required</p>
                    </div>
                </div>
                <Link to="/login" className="back-to-login" id="back-to-login" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <ArrowLeft size={18} /> Back to Login
                </Link>
            </div>

            <div className="emergency-tabs" id="emergency-tabs">
                <button
                    className={`emergency-tab ${activeTab === 'hospitals' ? 'active' : ''}`}
                    onClick={() => setActiveTab('hospitals')}
                    id="tab-hospitals"
                    style={{ display: 'flex', alignItems: 'center', gap: '10px' }}
                >
                    <Hospital size={20} /> Find Hospitals
                </button>
                <button
                    className={`emergency-tab ${activeTab === 'ambulance' ? 'active' : ''}`}
                    onClick={() => setActiveTab('ambulance')}
                    id="tab-ambulance"
                    style={{ display: 'flex', alignItems: 'center', gap: '10px' }}
                >
                    <Truck size={20} /> Track Ambulance
                </button>
            </div>

            <div className="emergency-content">
                {activeTab === 'hospitals' ? <HospitalFinder /> : <AmbulanceTracker />}
            </div>
        </div>
    )
}

export default Emergency
