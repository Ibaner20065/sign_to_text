import React, { useState, useEffect, useRef } from 'react'
import './ColorBlindFilter.css'

const filters = [
    { id: 'none', label: 'Standard Vision', short: '👁️', desc: 'Default color rendering' },
    { id: 'protanopia', label: 'Protanopia', short: '🔴', desc: 'Red-blind correction' },
    { id: 'deuteranopia', label: 'Deuteranopia', short: '🟢', desc: 'Green-blind correction' },
    { id: 'tritanopia', label: 'Tritanopia', short: '🔵', desc: 'Blue-blind correction' },
    { id: 'achromatopsia', label: 'Achromatopsia', short: '⚫', desc: 'Total color blindness' },
    { id: 'enhanced-contrast', label: 'Enhanced Contrast', short: '◐', desc: 'High contrast mode' },
]

const ColorBlindFilter = () => {
    const [open, setOpen] = useState(false)
    const [active, setActive] = useState('none')
    const menuRef = useRef(null)

    useEffect(() => {
        const saved = localStorage.getItem('auracare-color-filter')
        if (saved) {
            setActive(saved)
            applyFilter(saved)
        }
    }, [])

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) setOpen(false)
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const applyFilter = (id) => {
        const root = document.documentElement
        // Remove all filter classes
        filters.forEach(f => root.classList.remove(`filter-${f.id}`))
        // Apply selected
        if (id !== 'none') root.classList.add(`filter-${id}`)
    }

    const selectFilter = (id) => {
        setActive(id)
        applyFilter(id)
        localStorage.setItem('auracare-color-filter', id)
        setOpen(false)
    }

    return (
        <div className="cb-filter" ref={menuRef}>
            <button
                className="cb-filter-toggle"
                onClick={() => setOpen(!open)}
                aria-label="Color vision accessibility filters"
                title="Color Vision Filters"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20" />
                    <path d="M2 12h20" />
                </svg>
                {active !== 'none' && <span className="cb-filter-active-dot"></span>}
            </button>
            {open && (
                <div className="cb-filter-menu" role="menu" aria-label="Select color vision filter">
                    <div className="cb-filter-header">
                        <span>♿</span> Color Vision Accessibility
                    </div>
                    {filters.map(f => (
                        <button
                            key={f.id}
                            className={`cb-filter-option ${active === f.id ? 'active' : ''}`}
                            onClick={() => selectFilter(f.id)}
                            role="menuitem"
                        >
                            <span className="cb-filter-icon">{f.short}</span>
                            <div className="cb-filter-text">
                                <span className="cb-filter-label">{f.label}</span>
                                <span className="cb-filter-desc">{f.desc}</span>
                            </div>
                            {active === f.id && <span className="cb-filter-check">✓</span>}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}

export default ColorBlindFilter
