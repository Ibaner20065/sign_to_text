import React, { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Activity, Flame, Footprints, Droplets, Dumbbell, Zap, Heart, CheckCircle, TrendingUp, X } from 'lucide-react'
import './Health.css'

// ─── Helpers ──────────────────────────────────────────
const today = () => new Date().toISOString().slice(0, 10)
const dayLabel = (dateStr) => ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][new Date(dateStr).getDay()]

const loadData = (key, fallback) => {
    try { return JSON.parse(localStorage.getItem(key)) || fallback }
    catch { return fallback }
}
const saveData = (key, data) => localStorage.setItem(key, JSON.stringify(data))

const DEFAULT_DAILY = { steps: 0, calories: 0, water: 0, heartRate: 72 }
const DEFAULT_GOALS = { steps: 10000, water: 2500, calories: 500, exercise: 30 }

const EXERCISE_TYPES = [
    { type: 'Walking', icon: <Footprints size={20} />, met: 3.5, color: '#10b981' },
    { type: 'Running', icon: <Activity size={20} />, met: 8.0, color: '#ef4444' },
    { type: 'Yoga', icon: <Activity size={20} />, met: 2.5, color: '#8b5cf6' },
    { type: 'Physio', icon: <Activity size={20} />, met: 3.0, color: '#f59e0b' },
    { type: 'Gym', icon: <Dumbbell size={20} />, met: 6.0, color: '#3b82f6' },
    { type: 'Cycling', icon: <Activity size={20} />, met: 7.0, color: '#06b6d4' },
]

const Health = () => {
    const [dailyData, setDailyData] = useState(() => loadData(`auracare_health_${today()}`, DEFAULT_DAILY))
    const [exercises, setExercises] = useState(() => loadData(`auracare_exercises_${today()}`, []))
    const [goals] = useState(() => loadData('auracare_goals', DEFAULT_GOALS))
    const [streak, setStreak] = useState(() => loadData('auracare_streak', 0))
    const [weeklyData, setWeeklyData] = useState([])
    const [showModal, setShowModal] = useState(false)
    const [newExercise, setNewExercise] = useState({ type: 'Walking', duration: 15 })
    const [stepInput, setStepInput] = useState('')
    const [hrInput, setHrInput] = useState('')
    const [chartMetric, setChartMetric] = useState('steps')

    // Persist daily data
    useEffect(() => { saveData(`auracare_health_${today()}`, dailyData) }, [dailyData])
    useEffect(() => { saveData(`auracare_exercises_${today()}`, exercises) }, [exercises])

    // Build weekly data
    useEffect(() => {
        const data = []
        for (let i = 6; i >= 0; i--) {
            const d = new Date(); d.setDate(d.getDate() - i)
            const dateStr = d.toISOString().slice(0, 10)
            const stored = loadData(`auracare_health_${dateStr}`, null)
            data.push({
                day: dayLabel(dateStr),
                date: dateStr,
                steps: stored?.steps || 0,
                calories: stored?.calories || 0,
                water: stored?.water || 0,
            })
        }
        setWeeklyData(data)
    }, [dailyData])

    // Calculate streak
    useEffect(() => {
        let count = 0
        for (let i = 1; i <= 30; i++) {
            const d = new Date(); d.setDate(d.getDate() - i)
            const dateStr = d.toISOString().slice(0, 10)
            const stored = loadData(`auracare_health_${dateStr}`, null)
            if (stored && (stored.steps > 0 || stored.calories > 0)) count++
            else break
        }
        if (dailyData.steps > 0 || dailyData.calories > 0) count++
        setStreak(count)
        saveData('auracare_streak', count)
    }, [dailyData])

    const totalExerciseMinutes = exercises.reduce((sum, e) => sum + e.duration, 0)
    const totalExerciseCalories = exercises.reduce((sum, e) => sum + e.calories, 0)

    // Progress percentages
    const stepsProgress = Math.min((dailyData.steps / goals.steps) * 100, 100)
    const waterProgress = Math.min((dailyData.water / goals.water) * 100, 100)
    const caloriesProgress = Math.min(((dailyData.calories + totalExerciseCalories) / goals.calories) * 100, 100)
    const exerciseProgress = Math.min((totalExerciseMinutes / goals.exercise) * 100, 100)

    const addSteps = () => {
        const val = parseInt(stepInput)
        if (!val || val <= 0) return
        const cals = Math.round(val * 0.04)
        setDailyData(prev => ({ ...prev, steps: prev.steps + val, calories: prev.calories + cals }))
        setStepInput('')
    }

    const addWater = (ml) => {
        setDailyData(prev => ({ ...prev, water: prev.water + ml }))
    }

    const updateHeartRate = () => {
        const val = parseInt(hrInput)
        if (!val || val < 40 || val > 200) return
        setDailyData(prev => ({ ...prev, heartRate: val }))
        setHrInput('')
    }

    const addExercise = () => {
        const exerciseType = EXERCISE_TYPES.find(e => e.type === newExercise.type)
        const calories = Math.round(exerciseType.met * 70 * (newExercise.duration / 60))
        const entry = {
            id: Date.now(),
            type: newExercise.type,
            icon: exerciseType.icon,
            duration: newExercise.duration,
            calories,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        }
        setExercises(prev => [...prev, entry])
        setDailyData(prev => ({ ...prev, calories: prev.calories + calories }))
        setShowModal(false)
        setNewExercise({ type: 'Walking', duration: 15 })
    }

    const removeExercise = (id) => {
        const ex = exercises.find(e => e.id === id)
        if (ex) {
            setExercises(prev => prev.filter(e => e.id !== id))
            setDailyData(prev => ({ ...prev, calories: Math.max(0, prev.calories - ex.calories) }))
        }
    }

    const hrStatus = dailyData.heartRate < 60 ? 'low' : dailyData.heartRate > 100 ? 'high' : 'normal'

    return (
        <div id="health-page">
            <div className="health-header">
                <h1 className="page-title"><Activity size={32} style={{ verticalAlign: 'middle', marginRight: '12px' }} /> Health Monitor</h1>
                {streak > 0 && (
                    <div className="streak-badge" id="streak-badge" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Zap size={16} /> {streak} day streak
                    </div>
                )}
            </div>

            {/* ─── Stats Row ─── */}
            <div className="stats-grid" id="stats-grid">
                {/* Steps Card */}
                <div className="stat-card steps-card" id="steps-card">
                    <div className="stat-icon"><Footprints size={24} /></div>
                    <div className="stat-info">
                        <span className="stat-label">Steps</span>
                        <span className="stat-value">{dailyData.steps.toLocaleString()}</span>
                        <span className="stat-goal">/ {goals.steps.toLocaleString()}</span>
                    </div>
                    <div className="stat-progress">
                        <div className="progress-ring" style={{ '--progress': stepsProgress, '--ring-color': '#10b981' }}>
                            <span>{Math.round(stepsProgress)}%</span>
                        </div>
                    </div>
                    <div className="stat-action">
                        <input type="number" placeholder="Add steps" value={stepInput} onChange={e => setStepInput(e.target.value)} className="mini-input" id="step-input" onKeyDown={e => e.key === 'Enter' && addSteps()} />
                        <button onClick={addSteps} className="mini-btn" id="btn-add-steps">+</button>
                    </div>
                    {stepsProgress >= 100 && <span className="goal-complete">🟢 Goal Complete!</span>}
                    {stepsProgress < 50 && dailyData.steps > 0 && <span className="goal-low">🔴 Keep going!</span>}
                </div>

                {/* Calories Card */}
                <div className="stat-card calories-card" id="calories-card">
                    <div className="stat-icon"><Flame size={24} /></div>
                    <div className="stat-info">
                        <span className="stat-label">Calories</span>
                        <span className="stat-value">{(dailyData.calories + totalExerciseCalories).toLocaleString()}</span>
                        <span className="stat-goal">/ {goals.calories} kcal</span>
                    </div>
                    <div className="stat-progress">
                        <div className="progress-ring" style={{ '--progress': caloriesProgress, '--ring-color': '#f59e0b' }}>
                            <span>{Math.round(caloriesProgress)}%</span>
                        </div>
                    </div>
                    {caloriesProgress >= 100 && <span className="goal-complete">🟢 Goal Complete!</span>}
                </div>

                {/* Heart Rate Card */}
                <div className="stat-card heart-card" id="heart-card">
                    <div className="stat-icon heart-pulse"><Heart size={24} /></div>
                    <div className="stat-info">
                        <span className="stat-label">Heart Rate</span>
                        <span className="stat-value">{dailyData.heartRate} <small>bpm</small></span>
                        <span className={`hr-status ${hrStatus}`}>
                            {hrStatus === 'normal' ? '🟢 Normal' : hrStatus === 'low' ? '🔵 Low' : '🔴 High'}
                        </span>
                    </div>
                    <div className="stat-action">
                        <input type="number" placeholder="BPM" value={hrInput} onChange={e => setHrInput(e.target.value)} className="mini-input" id="hr-input" onKeyDown={e => e.key === 'Enter' && updateHeartRate()} />
                        <button onClick={updateHeartRate} className="mini-btn" id="btn-update-hr">✓</button>
                    </div>
                </div>

                {/* Water Card */}
                <div className="stat-card water-card" id="water-card">
                    <div className="stat-icon"><Droplets size={24} /></div>
                    <div className="stat-info">
                        <span className="stat-label">Water</span>
                        <span className="stat-value">{dailyData.water} <small>ml</small></span>
                        <span className="stat-goal">/ {goals.water} ml</span>
                    </div>
                    <div className="stat-progress">
                        <div className="progress-ring" style={{ '--progress': waterProgress, '--ring-color': '#3b82f6' }}>
                            <span>{Math.round(waterProgress)}%</span>
                        </div>
                    </div>
                    <div className="water-buttons">
                        <button onClick={() => addWater(250)} className="water-btn" id="btn-water-250">+250ml</button>
                        <button onClick={() => addWater(500)} className="water-btn" id="btn-water-500">+500ml</button>
                    </div>
                    {waterProgress >= 100 && <span className="goal-complete">🟢 Hydrated!</span>}
                </div>
            </div>

            {/* ─── Activity Progress Bars ─── */}
            <div className="card" id="activity-progress">
                <h2 className="section-title"><TrendingUp size={22} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Daily Progress</h2>
                <div className="progress-bars">
                    <div className="progress-item">
                        <div className="progress-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Footprints size={16} /> Steps</div>
                            <span>{dailyData.steps} / {goals.steps}</span>
                        </div>
                        <div className="progress-bar"><div className="progress-fill steps-fill" style={{ width: `${stepsProgress}%` }}></div></div>
                    </div>
                    <div className="progress-item">
                        <div className="progress-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Flame size={16} /> Calories</div>
                            <span>{dailyData.calories + totalExerciseCalories} / {goals.calories}</span>
                        </div>
                        <div className="progress-bar"><div className="progress-fill calories-fill" style={{ width: `${caloriesProgress}%` }}></div></div>
                    </div>
                    <div className="progress-item">
                        <div className="progress-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Droplets size={16} /> Water</div>
                            <span>{dailyData.water} / {goals.water} ml</span>
                        </div>
                        <div className="progress-bar"><div className="progress-fill water-fill" style={{ width: `${waterProgress}%` }}></div></div>
                    </div>
                    <div className="progress-item">
                        <div className="progress-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Dumbbell size={16} /> Exercise</div>
                            <span>{totalExerciseMinutes} / {goals.exercise} min</span>
                        </div>
                        <div className="progress-bar"><div className="progress-fill exercise-fill" style={{ width: `${exerciseProgress}%` }}></div></div>
                    </div>
                </div>
            </div>

            {/* ─── Bottom Grid: Exercise Log + Weekly Chart ─── */}
            <div className="health-bottom-grid">
                {/* Exercise Log */}
                <div className="card" id="exercise-log">
                    <div className="section-header">
                        <h2 className="section-title"><Dumbbell size={22} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Exercise Log</h2>
                        <button className="button button-primary" onClick={() => setShowModal(true)} id="btn-add-exercise">
                            + Add Exercise
                        </button>
                    </div>
                    {exercises.length === 0 ? (
                        <p className="empty-state">No exercises logged today. Start moving! <Activity size={18} style={{ verticalAlign: 'middle' }} /></p>
                    ) : (
                        <div className="exercise-list">
                            {exercises.map(ex => (
                                <div key={ex.id} className="exercise-item" id={`exercise-${ex.id}`}>
                                    <span className="exercise-icon">{ex.icon}</span>
                                    <div className="exercise-details">
                                        <strong>{ex.type}</strong>
                                        <span>{ex.duration} min • {ex.calories} kcal • {ex.time}</span>
                                    </div>
                                    <button className="remove-btn" onClick={() => removeExercise(ex.id)} aria-label="Remove"><X size={16} /></button>
                                </div>
                            ))}
                            <div className="exercise-summary">
                                Total: <strong>{totalExerciseMinutes} min</strong> • <strong>{totalExerciseCalories} kcal</strong>
                            </div>
                        </div>
                    )}
                </div>

                {/* Weekly Chart */}
                <div className="card" id="weekly-chart">
                    <div className="section-header">
                        <h2 className="section-title"><TrendingUp size={22} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Weekly Activity</h2>
                        <div className="chart-toggle">
                            <button className={`toggle-btn ${chartMetric === 'steps' ? 'active' : ''}`} onClick={() => setChartMetric('steps')}>Steps</button>
                            <button className={`toggle-btn ${chartMetric === 'calories' ? 'active' : ''}`} onClick={() => setChartMetric('calories')}>Calories</button>
                            <button className={`toggle-btn ${chartMetric === 'water' ? 'active' : ''}`} onClick={() => setChartMetric('water')}>Water</button>
                        </div>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height={260}>
                            <BarChart data={weeklyData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                                <XAxis dataKey="day" tick={{ fill: 'var(--text-secondary)', fontSize: 13 }} />
                                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{ background: 'white', border: '1px solid var(--border-light)', borderRadius: 8, fontSize: '0.85rem' }}
                                />
                                <Bar
                                    dataKey={chartMetric}
                                    fill={chartMetric === 'steps' ? '#10b981' : chartMetric === 'calories' ? '#f59e0b' : '#3b82f6'}
                                    radius={[6, 6, 0, 0]}
                                    maxBarSize={40}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* ─── Add Exercise Modal ─── */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()} id="add-exercise-modal">
                        <h2>Add Exercise</h2>
                        <div className="exercise-type-grid">
                            {EXERCISE_TYPES.map(et => (
                                <button
                                    key={et.type}
                                    className={`exercise-type-btn ${newExercise.type === et.type ? 'selected' : ''}`}
                                    onClick={() => setNewExercise(prev => ({ ...prev, type: et.type }))}
                                    style={{ '--sel-color': et.color }}
                                >
                                    <span className="et-icon">{et.icon}</span>
                                    <span className="et-label">{et.type}</span>
                                </button>
                            ))}
                        </div>
                        <div className="form-group">
                            <label className="label">Duration (minutes)</label>
                            <input
                                type="number"
                                className="input"
                                value={newExercise.duration}
                                onChange={e => setNewExercise(prev => ({ ...prev, duration: parseInt(e.target.value) || 0 }))}
                                min="1"
                                max="300"
                                id="exercise-duration"
                            />
                        </div>
                        <p className="calorie-preview">
                            Est. burn: <strong>{Math.round((EXERCISE_TYPES.find(e => e.type === newExercise.type)?.met || 3) * 70 * (newExercise.duration / 60))} kcal</strong>
                        </p>
                        <div className="modal-buttons">
                            <button className="button button-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                            <button className="button button-primary" onClick={addExercise} disabled={newExercise.duration <= 0} id="btn-save-exercise">
                                Save Exercise
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default Health
