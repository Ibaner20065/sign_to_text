import React, { useState, useRef, useEffect } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils'

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
]
import './Communication.css'

// ────────────────────────────────────────────
// Gesture Recognition Engine (client-side)
// ────────────────────────────────────────────

function distance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
}

function distance2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

/**
 * Determine if each finger is extended.
 * Returns [thumb, index, middle, ring, pinky] as booleans.
 * MediaPipe landmarks: wrist=0, thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
 */
function getFingerStates(lm) {
  // For fingers (not thumb): tip is above (lower y) PIP joint = extended
  const indexExtended = lm[8].y < lm[6].y
  const middleExtended = lm[12].y < lm[10].y
  const ringExtended = lm[16].y < lm[14].y
  const pinkyExtended = lm[20].y < lm[18].y

  // Thumb: compare tip x vs IP joint x — depends on handedness
  // If thumb tip is further from palm center than IP joint, it's extended
  const palmCenterX = (lm[0].x + lm[9].x) / 2
  const thumbTipDist = Math.abs(lm[4].x - palmCenterX)
  const thumbIPDist = Math.abs(lm[3].x - palmCenterX)
  const thumbExtended = thumbTipDist > thumbIPDist

  return [thumbExtended, indexExtended, middleExtended, ringExtended, pinkyExtended]
}

function countExtendedFingers(fingerStates) {
  return fingerStates.filter(Boolean).length
}

/**
 * Classify a hand gesture from 21 MediaPipe landmarks.
 * Returns { gesture: string, confidence: number } or null.
 */
function classifyGesture(landmarks) {
  const lm = landmarks
  const fingers = getFingerStates(lm)
  const [thumb, index, middle, ring, pinky] = fingers
  const extCount = countExtendedFingers(fingers)

  // Calculate some useful metrics
  const palmSize = distance2D(lm[0], lm[9])

  // Finger spread: distance between index tip and pinky tip
  const fingerSpread = distance2D(lm[8], lm[20]) / palmSize

  // Hand openness: average distance from fingertips to wrist
  const avgTipDist = ([8, 12, 16, 20].reduce((s, i) => s + distance2D(lm[i], lm[0]), 0)) / 4 / palmSize

  // ── HELLO / OPEN PALM ──
  // All 5 fingers extended with spread
  if (extCount >= 4 && fingerSpread > 0.8 && avgTipDist > 1.3) {
    // Check if fingers are spread (open hand wave)
    return { gesture: 'hello', confidence: 0.7 + Math.min(0.25, (fingerSpread - 0.8) * 0.5) }
  }

  // ── YES / THUMBS UP ──
  // Only thumb extended, all other fingers curled
  if (thumb && !index && !middle && !ring && !pinky) {
    // Thumb should be pointing upward (thumb tip y < thumb IP y significantly)
    const thumbUpward = lm[4].y < lm[3].y && lm[4].y < lm[2].y
    if (thumbUpward) {
      return { gesture: 'yes', confidence: 0.92 }
    }
    // Thumb out but not pointing up
    return { gesture: 'yes', confidence: 0.75 }
  }

  // ── NO / CLOSED FIST ──
  // No fingers extended (or just barely)
  if (extCount === 0) {
    return { gesture: 'no', confidence: 0.90 }
  }

  // ── I / PINKY ONLY ──
  // Only pinky finger extended (ASL letter I)
  if (pinky && !index && !middle && !ring && !thumb) {
    return { gesture: 'I', confidence: 0.90 }
  }

  // ── YOU / POINTING ──
  // Only index finger extended (pointing at someone)
  if (index && !middle && !ring && !pinky && !thumb) {
    return { gesture: 'you', confidence: 0.88 }
  }

  // ── ME / POINTING TO SELF ──
  // Index finger extended + pointing toward camera (z-depth check)
  // When pointing at self: index tip z is more negative (closer to camera)
  if (index && !middle && !ring && !pinky && thumb) {
    // Thumb + index = could be "ME" (thumb naturally sticks out when pointing at self)
    return { gesture: 'ME', confidence: 0.80 }
  }

  // ── PEACE / TWO FINGERS ──
  // Index + middle extended, rest closed (V sign)
  if (index && middle && !ring && !pinky) {
    return { gesture: 'peace', confidence: 0.85 }
  }

  // ── THREE FINGERS ──
  if (index && middle && ring && !pinky && !thumb) {
    return { gesture: 'three', confidence: 0.80 }
  }

  // ── STOP / FLAT PALM FORWARD ──
  // All fingers extended but close together (not spread like hello)
  if (extCount >= 4 && fingerSpread < 0.8) {
    return { gesture: 'stop', confidence: 0.75 }
  }

  // ── HELP (fist on open palm — two hands needed, fallback) ──
  // With one hand: thumb + pinky extended, rest curled (ASL "Y" / call-me)
  if (thumb && pinky && !index && !middle && !ring) {
    return { gesture: 'help', confidence: 0.78 }
  }

  return null
}

// ────────────────────────────────────────────
// Communication Component
// ────────────────────────────────────────────

const Communication = () => {
  const [mode, setMode] = useState('sign')
  const [sentence, setSentence] = useState('')
  const [currentWord, setCurrentWord] = useState('')
  const [currentConfidence, setCurrentConfidence] = useState(0)
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [showCopyToast, setShowCopyToast] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [speechStatus, setSpeechStatus] = useState('')
  const [showGuide, setShowGuide] = useState(false)
  const [noSpeechTimer, setNoSpeechTimer] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)
  const predictionBufferRef = useRef([])
  const lastWordRef = useRef(null)
  const recognitionRef = useRef(null)
  const noHandFramesRef = useRef(0)

  const PREDICTION_WINDOW = 15
  const CONFIDENCE_THRESHOLD = 0.55
  const GESTURE_HOLD_FRAMES = 10 // Frames of consistent gesture before accepting

  useEffect(() => {
    if (mode === 'sign') {
      initializeMediaPipe()
    } else {
      cleanupMediaPipe()
    }

    return () => {
      cleanupMediaPipe()
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [mode])

  const initializeMediaPipe = () => {
    if (!videoRef.current || !canvasRef.current) return

    const setCanvasSize = () => {
      if (videoRef.current && canvasRef.current) {
        canvasRef.current.width = videoRef.current.videoWidth || 640
        canvasRef.current.height = videoRef.current.videoHeight || 480
      }
    }

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      },
    })

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })

    hands.onResults(onResults)

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        setCanvasSize()
        await hands.send({ image: videoRef.current })
      },
      width: 640,
      height: 480,
    })

    camera.start()
    setCameraActive(true)
    handsRef.current = hands
    cameraRef.current = camera
  }

  const cleanupMediaPipe = () => {
    if (cameraRef.current) {
      cameraRef.current.stop()
      cameraRef.current = null
    }
    handsRef.current = null
    setCameraActive(false)
  }

  const onResults = async (results) => {
    const canvasCtx = canvasRef.current.getContext('2d')
    canvasCtx.save()
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const handLandmarks = results.multiHandLandmarks[0]
      noHandFramesRef.current = 0

      drawConnectors(canvasCtx, handLandmarks, HAND_CONNECTIONS, {
        color: '#2dd4bf',
        lineWidth: 2
      })
      drawLandmarks(canvasCtx, handLandmarks, {
        color: '#0d86e4',
        lineWidth: 1,
        radius: 3
      })

      try {
        // Real gesture classification
        const result = classifyGesture(handLandmarks)

        if (result && result.confidence >= 0.65) {
          setCurrentWord(result.gesture)
          setCurrentConfidence(Math.round(result.confidence * 100))
          predictionBufferRef.current.push(result.gesture)

          // Draw gesture label on canvas
          canvasCtx.font = 'bold 28px Inter, sans-serif'
          canvasCtx.fillStyle = '#2dd4bf'
          canvasCtx.strokeStyle = 'rgba(0,0,0,0.7)'
          canvasCtx.lineWidth = 4
          canvasCtx.strokeText(`${result.gesture} (${Math.round(result.confidence * 100)}%)`, 20, 40)
          canvasCtx.fillText(`${result.gesture} (${Math.round(result.confidence * 100)}%)`, 20, 40)

          if (predictionBufferRef.current.length >= PREDICTION_WINDOW) {
            const nonEmpty = predictionBufferRef.current.filter((p) => p !== '')
            if (nonEmpty.length > 0) {
              const counts = {}
              nonEmpty.forEach((p) => {
                counts[p] = (counts[p] || 0) + 1
              })

              const mostCommon = Object.keys(counts).reduce((a, b) =>
                counts[a] > counts[b] ? a : b
              )
              const count = counts[mostCommon]

              if (count >= PREDICTION_WINDOW * CONFIDENCE_THRESHOLD) {
                if (mostCommon !== lastWordRef.current) {
                  lastWordRef.current = mostCommon
                  setSentence((prev) => (prev ? `${prev} ${mostCommon}` : mostCommon))
                  predictionBufferRef.current = []
                }
              }
            }

            // Keep buffer from growing infinitely
            if (predictionBufferRef.current.length > PREDICTION_WINDOW * 2) {
              predictionBufferRef.current = predictionBufferRef.current.slice(-PREDICTION_WINDOW)
            }
          }
        } else {
          setCurrentWord('')
          setCurrentConfidence(0)
        }
      } catch (error) {
        console.error('Prediction error:', error)
      }
    } else {
      // No hand detected
      noHandFramesRef.current++
      if (noHandFramesRef.current > 30) {
        // Reset after ~1 second of no hand
        setCurrentWord('')
        setCurrentConfidence(0)
        lastWordRef.current = null
        predictionBufferRef.current = []
      }
    }

    canvasCtx.restore()
  }

  const handleSpeak = () => {
    if (sentence && 'speechSynthesis' in window) {
      const savedSettings = localStorage.getItem('userSettings')
      let speed = 1.0
      if (savedSettings) {
        const settings = JSON.parse(savedSettings)
        speed = settings.ttsSpeed || 1.0
      }
      const utterance = new SpeechSynthesisUtterance(sentence)
      utterance.rate = speed
      window.speechSynthesis.speak(utterance)
    }
  }

  const handleClear = () => {
    setSentence('')
    setCurrentWord('')
    setCurrentConfidence(0)
    predictionBufferRef.current = []
    lastWordRef.current = null
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(sentence)
    setShowCopyToast(true)
    setTimeout(() => setShowCopyToast(false), 2000)
  }

  // ── Speech Recognition ──

  const finalTranscriptRef = useRef('')
  const shouldRestartRef = useRef(false)
  const speechTimeoutRef = useRef(null)

  const startSpeechRecognition = () => {
    // Check for secure context
    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      setSpeechStatus('⚠️ Speech recognition requires HTTPS or localhost. Please use a secure connection.')
      return
    }

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setSpeechStatus('❌ Speech recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    finalTranscriptRef.current = transcript
    shouldRestartRef.current = true

    setSpeechStatus('🎤 Listening... Speak now!')

    // Set a timeout to warn if no speech detected
    speechTimeoutRef.current = setTimeout(() => {
      if (isRecording && !transcript) {
        setSpeechStatus('🔇 No speech detected yet. Make sure your microphone is working and speak clearly.')
      }
    }, 5000)

    recognition.onresult = (event) => {
      // Clear the no-speech timeout
      if (speechTimeoutRef.current) {
        clearTimeout(speechTimeoutRef.current)
        speechTimeoutRef.current = null
      }

      let interimTranscript = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscriptRef.current += t + ' '
          setSpeechStatus('🎤 Listening...')
        } else {
          interimTranscript += t
          setSpeechStatus('🎤 Hearing you...')
        }
      }

      setTranscript(finalTranscriptRef.current + interimTranscript)
    }

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      if (speechTimeoutRef.current) {
        clearTimeout(speechTimeoutRef.current)
        speechTimeoutRef.current = null
      }

      if (event.error === 'not-allowed') {
        setSpeechStatus('❌ Microphone access denied. Please allow microphone permission in your browser settings.')
        shouldRestartRef.current = false
      } else if (event.error === 'network') {
        setSpeechStatus('❌ Network error. Speech recognition requires an internet connection (uses Google servers).')
        shouldRestartRef.current = false
      } else if (event.error === 'no-speech') {
        setSpeechStatus('🔇 No speech detected. Try speaking louder or closer to the mic.')
        // Don't stop — let it restart
      } else if (event.error === 'audio-capture') {
        setSpeechStatus('❌ No microphone found. Please connect a microphone and try again.')
        shouldRestartRef.current = false
      } else if (event.error !== 'aborted') {
        setSpeechStatus(`⚠️ Error: ${event.error}. Retrying...`)
      }
    }

    recognition.onend = () => {
      // Chrome stops recognition after silence — auto-restart for continuous captioning
      if (shouldRestartRef.current) {
        try {
          recognition.start()
          setSpeechStatus('🎤 Listening...')
        } catch (e) {
          console.error('Failed to restart recognition:', e)
          setIsRecording(false)
          shouldRestartRef.current = false
          setSpeechStatus('❌ Recognition stopped unexpectedly. Click Start to try again.')
        }
      } else {
        setIsRecording(false)
        setSpeechStatus('')
      }
    }

    try {
      recognition.start()
      recognitionRef.current = recognition
      setIsRecording(true)
    } catch (e) {
      console.error('Failed to start recognition:', e)
      setSpeechStatus('❌ Could not start speech recognition. Is another tab using the microphone?')
    }
  }

  const stopSpeechRecognition = () => {
    shouldRestartRef.current = false
    if (speechTimeoutRef.current) {
      clearTimeout(speechTimeoutRef.current)
      speechTimeoutRef.current = null
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
      setIsRecording(false)
      setSpeechStatus('')
    }
  }

  const clearTranscript = () => {
    setTranscript('')
    finalTranscriptRef.current = ''
  }

  // ── Gesture guide data ──
  const gestureGuide = [
    { gesture: 'hello', icon: '👋', desc: 'Open palm, all fingers spread' },
    { gesture: 'yes', icon: '👍', desc: 'Thumbs up, fist closed' },
    { gesture: 'no', icon: '✊', desc: 'Closed fist, no fingers out' },
    { gesture: 'I', icon: '🤙', desc: 'Only pinky finger extended' },
    { gesture: 'you', icon: '👆', desc: 'Only index finger pointing' },
    { gesture: 'ME', icon: '👈', desc: 'Index + thumb out (point at self)' },
    { gesture: 'peace', icon: '✌️', desc: 'Index + middle fingers (V sign)' },
    { gesture: 'stop', icon: '✋', desc: 'Palm forward, fingers together' },
    { gesture: 'help', icon: '🤙', desc: 'Thumb + pinky extended (hang loose)' },
  ]

  return (
    <div id="communication-page">
      <h1 className="page-title">🤟 Communication</h1>

      <div className="mode-toggle" id="mode-toggle">
        <button
          className={`mode-button ${mode === 'sign' ? 'active' : ''}`}
          onClick={() => setMode('sign')}
          id="mode-sign"
        >
          ✋ Sign to Speech
        </button>
        <button
          className={`mode-button ${mode === 'speech' ? 'active' : ''}`}
          onClick={() => setMode('speech')}
          id="mode-speech"
        >
          🎤 Speech to Text
        </button>
      </div>

      {mode === 'sign' ? (
        <div className="sign-mode">
          <div className="card">
            <div className="video-container" id="webcam-container">
              <video ref={videoRef} className="video" autoPlay playsInline></video>
              <canvas ref={canvasRef} className="canvas"></canvas>
              <div className="webcam-status">
                <span className={`status-dot ${cameraActive ? '' : 'inactive'}`}></span>
                <span>{cameraActive ? 'Camera Active' : 'Initializing...'}</span>
              </div>
            </div>

            {/* Current Detection */}
            <div className="prediction-display">
              <div className="current-word" id="current-word">
                {currentWord ? (
                  <>
                    Detected: <strong>{currentWord}</strong>
                    <span className="confidence-badge">{currentConfidence}%</span>
                  </>
                ) : (
                  'Show a sign to the camera...'
                )}
              </div>

              {/* Confidence Bar */}
              {currentWord && (
                <div className="confidence-bar-container">
                  <div
                    className="confidence-bar"
                    style={{
                      width: `${currentConfidence}%`,
                      background: currentConfidence > 80 ? '#2dd4bf' : currentConfidence > 60 ? '#f59e0b' : '#ef4444'
                    }}
                  />
                </div>
              )}

              <div className="sentence-display" id="sentence-display">
                <h3>Recognized Sentence</h3>
                <p className={`sentence-text ${!sentence ? 'empty' : ''}`}>
                  {sentence || 'Show signs to the camera to build a sentence...'}
                </p>
              </div>
              <div className="controls">
                <button className="button button-primary" onClick={handleSpeak} disabled={!sentence} id="btn-speak">
                  🔊 Speak
                </button>
                <button className="button button-secondary" onClick={handleClear} id="btn-clear-sign">
                  🗑️ Clear
                </button>
                <button className="button button-secondary" onClick={handleCopy} disabled={!sentence} id="btn-copy">
                  📋 Copy
                </button>
                <button
                  className="button button-secondary"
                  onClick={() => setShowGuide(!showGuide)}
                  id="btn-guide"
                >
                  {showGuide ? '✕ Hide Guide' : '📖 Gesture Guide'}
                </button>
              </div>
            </div>

            {/* Gesture Guide */}
            {showGuide && (
              <div className="gesture-guide" id="gesture-guide">
                <h3>Supported Gestures</h3>
                <div className="gesture-grid">
                  {gestureGuide.map((g) => (
                    <div className="gesture-item" key={g.gesture}>
                      <span className="gesture-icon">{g.icon}</span>
                      <strong>{g.gesture}</strong>
                      <span className="gesture-desc">{g.desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="speech-mode">
          <div className="card">
            <h3>Live Captioning</h3>
            <p className="description">
              Speak into your microphone. Your words will be transcribed in real-time so a deaf user can read along.
            </p>

            {/* Status message */}
            {speechStatus && (
              <div className={`speech-status ${speechStatus.startsWith('❌') ? 'error' : speechStatus.startsWith('⚠️') ? 'warning' : 'info'}`} id="speech-status">
                {speechStatus}
              </div>
            )}

            {isRecording && (
              <div className="recording-indicator" id="recording-indicator">
                <span className="recording-dot"></span>
                Recording...
              </div>
            )}

            <div className="transcript-container" id="transcript-container">
              <div className={`transcript-text ${!transcript ? 'empty' : ''}`}>
                {transcript || '🎤 Press "Start Recording" to begin live captioning...'}
              </div>
            </div>
            <div className="controls">
              {!isRecording ? (
                <button className="button button-primary" onClick={startSpeechRecognition} id="btn-start-recording">
                  🎤 Start Recording
                </button>
              ) : (
                <button className="button button-danger" onClick={stopSpeechRecognition} id="btn-stop-recording">
                  ⏹️ Stop Recording
                </button>
              )}
              <button className="button button-secondary" onClick={clearTranscript} id="btn-clear-transcript">
                🗑️ Clear
              </button>
            </div>
          </div>
        </div>
      )}

      {showCopyToast && (
        <div className="copy-toast" id="copy-toast">
          ✅ Copied to clipboard!
        </div>
      )}
    </div>
  )
}

export default Communication
