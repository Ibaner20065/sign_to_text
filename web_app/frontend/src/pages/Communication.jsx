import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils'
import { Hand, Mic, Volume2, Trash2, Clipboard, BookOpen, X, CheckCircle, Info, AlertTriangle, AlertOctagon, MicOff, Square, MessageSquare, ThumbsUp, MousePointer2, User, HandMetal, Play, Pause, SkipForward } from 'lucide-react'

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
// Speech-to-Sign: ASL Data & Mappings
// ────────────────────────────────────────────

const ASL_ALPHABET = {
  a: { hand: '🤛', desc: 'Fist, thumb beside index finger' },
  b: { hand: '🖐', desc: 'Flat hand, fingers together, thumb tucked in palm' },
  c: { hand: '🫲', desc: 'Curved hand, like holding a small ball' },
  d: { hand: '☝️', desc: 'Index up, others touch thumb in circle' },
  e: { hand: '🤛', desc: 'Fingers curled into palm, thumb tucked under' },
  f: { hand: '👌', desc: 'Index & thumb circle, other 3 fingers up' },
  g: { hand: '👈', desc: 'Index & thumb point sideways, fist closed' },
  h: { hand: '👈', desc: 'Index & middle point sideways together' },
  i: { hand: '🤙', desc: 'Pinky up, other fingers in fist' },
  j: { hand: '🤙', desc: 'Pinky up, trace a J motion downward' },
  k: { hand: '✌️', desc: 'Index & middle up in V, thumb between them' },
  l: { hand: '🤟', desc: 'L-shape: index up, thumb out, others curled' },
  m: { hand: '🤛', desc: 'Three fingers over thumb in fist' },
  n: { hand: '🤛', desc: 'Two fingers over thumb in fist' },
  o: { hand: '🫰', desc: 'All fingertips touch thumb, forming O' },
  p: { hand: '👇', desc: 'Like K but pointing downward' },
  q: { hand: '👇', desc: 'Like G but pointing downward' },
  r: { hand: '🤞', desc: 'Index & middle crossed, others in fist' },
  s: { hand: '✊', desc: 'Fist with thumb over fingers' },
  t: { hand: '🤛', desc: 'Thumb tucked between index & middle' },
  u: { hand: '✌️', desc: 'Index & middle up together, others curled' },
  v: { hand: '✌️', desc: 'Index & middle up spread apart (V sign)' },
  w: { hand: '🤟', desc: 'Index, middle & ring up spread, others curled' },
  x: { hand: '☝️', desc: 'Index finger bent like a hook' },
  y: { hand: '🤙', desc: 'Thumb & pinky out (hang loose)' },
  z: { hand: '☝️', desc: 'Index finger traces Z shape in air' },
}

const KNOWN_SIGNS = {
  hello: { emoji: '👋', desc: 'Open hand near forehead, wave outward', color: '#2dd4bf' },
  hi: { emoji: '👋', desc: 'Open hand near forehead, wave outward', color: '#2dd4bf' },
  hey: { emoji: '👋', desc: 'Open hand near forehead, wave outward', color: '#2dd4bf' },
  yes: { emoji: '👍', desc: 'Make fist, nod it up and down (like nodding)', color: '#22c55e' },
  yeah: { emoji: '👍', desc: 'Make fist, nod it up and down', color: '#22c55e' },
  no: { emoji: '✊', desc: 'Snap index+middle against thumb (like closing)', color: '#ef4444' },
  nope: { emoji: '✊', desc: 'Snap index+middle against thumb', color: '#ef4444' },
  you: { emoji: '👉', desc: 'Point index finger at the person', color: '#3b82f6' },
  me: { emoji: '👈', desc: 'Point index finger at your own chest', color: '#8b5cf6' },
  i: { emoji: '🤙', desc: 'Pinky finger up, point at yourself', color: '#8b5cf6' },
  please: { emoji: '🤚', desc: 'Flat hand circles on chest', color: '#f59e0b' },
  thanks: { emoji: '🤚', desc: 'Fingertips on chin, move hand outward', color: '#f59e0b' },
  'thank you': { emoji: '🤚', desc: 'Fingertips on chin, move hand outward', color: '#f59e0b' },
  sorry: { emoji: '✊', desc: 'Fist circles on chest', color: '#f97316' },
  help: { emoji: '👐', desc: 'Fist on open palm, lift both up', color: '#ef4444' },
  stop: { emoji: '🤚', desc: 'Flat hand chops into open palm', color: '#ef4444' },
  love: { emoji: '🤟', desc: 'Thumb + index + pinky extended (ILY sign)', color: '#ec4899' },
  good: { emoji: '👍', desc: 'Flat hand from chin, drops into open palm', color: '#22c55e' },
  bad: { emoji: '👎', desc: 'Flat hand from chin, flip palm down', color: '#ef4444' },
  friend: { emoji: '🤝', desc: 'Hook index fingers together, flip', color: '#3b82f6' },
  water: { emoji: '💧', desc: 'W-hand taps chin twice', color: '#06b6d4' },
  food: { emoji: '🍽️', desc: 'Pinched fingers tap mouth', color: '#f59e0b' },
  eat: { emoji: '🍽️', desc: 'Pinched fingers tap mouth', color: '#f59e0b' },
  drink: { emoji: '🥤', desc: 'C-hand tilts toward mouth', color: '#06b6d4' },
  home: { emoji: '🏠', desc: 'Flat O on cheek, moves to jaw', color: '#8b5cf6' },
  hospital: { emoji: '🏥', desc: 'H-hand traces cross on upper arm', color: '#ef4444' },
  doctor: { emoji: '⚕️', desc: 'D-hand taps wrist pulse point', color: '#22c55e' },
  pain: { emoji: '😣', desc: 'Index fingers point at each other, twist', color: '#ef4444' },
  hurt: { emoji: '😣', desc: 'Index fingers point at each other, twist', color: '#ef4444' },
  medicine: { emoji: '💊', desc: 'Middle finger circles on open palm', color: '#8b5cf6' },
  emergency: { emoji: '🚨', desc: 'E-hand shakes side to side rapidly', color: '#ef4444' },
  ambulance: { emoji: '🚑', desc: 'Cross on arm + mime driving', color: '#ef4444' },
  name: { emoji: '✌️', desc: 'H-fingers tap on other H-fingers (crossed)', color: '#3b82f6' },
  what: { emoji: '🤷', desc: 'Palms up, shake side to side', color: '#f59e0b' },
  where: { emoji: '👆', desc: 'Index finger wags side to side', color: '#f59e0b' },
  when: { emoji: '🔄', desc: 'Index circles other index, then point forward', color: '#f59e0b' },
  how: { emoji: '🤲', desc: 'Knuckles together, roll hands out & up', color: '#f59e0b' },
  why: { emoji: '🤔', desc: 'Touch forehead, bring hand down to Y-shape', color: '#f59e0b' },
}

/**
 * Convert a word to its sign representation.
 * Returns either a known sign or an array of fingerspelled letters.
 */
function wordToSign(word) {
  const lower = word.toLowerCase().replace(/[^a-z]/g, '')
  if (!lower) return null

  if (KNOWN_SIGNS[lower]) {
    return { type: 'sign', word: lower, ...KNOWN_SIGNS[lower] }
  }

  // Fingerspell the word
  const letters = lower.split('').map(ch => ({
    letter: ch,
    ...(ASL_ALPHABET[ch] || { hand: '❓', desc: 'Unknown character' })
  }))

  return { type: 'fingerspell', word: lower, letters }
}

/**
 * Convert a full transcript into sign tokens.
 */
function transcriptToSigns(text) {
  if (!text || !text.trim()) return []
  const words = text.trim().split(/\s+/)
  return words.map(w => wordToSign(w)).filter(Boolean)
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
  const [signTokens, setSignTokens] = useState([])
  const [activeSignIndex, setActiveSignIndex] = useState(-1)
  const [activeLetterIndex, setActiveLetterIndex] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState(1200) // ms per sign/letter
  const signAnimationRef = useRef(null)
  const signDisplayRef = useRef(null)
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
    setSignTokens([])
    setActiveSignIndex(-1)
    setActiveLetterIndex(0)
    stopSignAnimation()
  }

  // ── Sign Animation Logic ──

  // Update sign tokens whenever transcript changes
  useEffect(() => {
    if (mode === 'speech' && transcript) {
      const tokens = transcriptToSigns(transcript)
      setSignTokens(tokens)
    }
  }, [transcript, mode])

  const stopSignAnimation = useCallback(() => {
    setIsAnimating(false)
    if (signAnimationRef.current) {
      clearInterval(signAnimationRef.current)
      signAnimationRef.current = null
    }
  }, [])

  const playSignAnimation = useCallback(() => {
    if (signTokens.length === 0) return

    stopSignAnimation()
    setIsAnimating(true)
    setActiveSignIndex(0)
    setActiveLetterIndex(0)

    let wordIdx = 0
    let letterIdx = 0

    signAnimationRef.current = setInterval(() => {
      const token = signTokens[wordIdx]
      if (!token) {
        // Reached the end
        stopSignAnimation()
        setActiveSignIndex(signTokens.length - 1)
        return
      }

      if (token.type === 'fingerspell') {
        // Advance through letters
        if (letterIdx < token.letters.length - 1) {
          letterIdx++
          setActiveLetterIndex(letterIdx)
        } else {
          // Move to next word
          wordIdx++
          letterIdx = 0
          setActiveSignIndex(wordIdx)
          setActiveLetterIndex(0)
        }
      } else {
        // Known sign — hold for a beat then advance
        wordIdx++
        letterIdx = 0
        setActiveSignIndex(wordIdx)
        setActiveLetterIndex(0)
      }

      // Scroll active sign into view
      if (signDisplayRef.current) {
        const activeEl = signDisplayRef.current.querySelector('.sign-card.active')
        if (activeEl) {
          activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
        }
      }
    }, animationSpeed)
  }, [signTokens, animationSpeed, stopSignAnimation])

  const skipToNextSign = () => {
    if (activeSignIndex < signTokens.length - 1) {
      setActiveSignIndex(prev => prev + 1)
      setActiveLetterIndex(0)
    }
  }

  // Cleanup animation on unmount or mode change
  useEffect(() => {
    return () => stopSignAnimation()
  }, [mode, stopSignAnimation])

  // ── Gesture guide data ──
  const gestureGuide = [
    { gesture: 'hello', icon: <Hand size={24} />, desc: 'Open palm, all fingers spread' },
    { gesture: 'yes', icon: <ThumbsUp size={24} />, desc: 'Thumbs up, fist closed' },
    { gesture: 'no', icon: <Hand size={24} />, desc: 'Closed fist, no fingers out' },
    { gesture: 'I', icon: <MessageSquare size={24} />, desc: 'Only pinky finger extended' },
    { gesture: 'you', icon: <MousePointer2 size={24} />, desc: 'Only index finger pointing' },
    { gesture: 'ME', icon: <User size={24} />, desc: 'Index + thumb out (point at self)' },
    { gesture: 'peace', icon: <HandMetal size={24} />, desc: 'Index + middle fingers (V sign)' },
    { gesture: 'stop', icon: <Hand size={24} />, desc: 'Palm forward, fingers together' },
    { gesture: 'help', icon: <AlertTriangle size={24} />, desc: 'Thumb + pinky extended (hang loose)' },
  ]

  return (
    <div id="communication-page">
      <h1 className="page-title"><Hand size={36} style={{ verticalAlign: 'middle', marginRight: '12px' }} /> Communication</h1>

      <div className="mode-toggle" id="mode-toggle">
        <button
          className={`mode-button ${mode === 'sign' ? 'active' : ''}`}
          onClick={() => setMode('sign')}
          id="mode-sign"
        >
          <Hand size={18} /> Sign to Speech
        </button>
        <button
          className={`mode-button ${mode === 'speech' ? 'active' : ''}`}
          onClick={() => setMode('speech')}
          id="mode-speech"
        >
          <Mic size={18} /> Speech to Sign
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
                  <Volume2 size={18} /> Speak
                </button>
                <button className="button button-secondary" onClick={handleClear} id="btn-clear-sign">
                  <Trash2 size={18} /> Clear
                </button>
                <button className="button button-secondary" onClick={handleCopy} disabled={!sentence} id="btn-copy">
                  <Clipboard size={18} /> Copy
                </button>
                <button
                  className="button button-secondary"
                  onClick={() => setShowGuide(!showGuide)}
                  id="btn-guide"
                >
                  {showGuide ? <><X size={18} /> Hide Guide</> : <><BookOpen size={18} /> Gesture Guide</>}
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
            <h3><Hand size={22} style={{ verticalAlign: 'middle', marginRight: '8px' }} />Speech to Sign Language</h3>
            <p className="description">
              Speak into your microphone. Your words will be converted into sign language visuals in real-time.
            </p>

            {/* Status message */}
            {speechStatus && (
              <div className={`speech-status ${speechStatus.includes('❌') ? 'error' : speechStatus.includes('⚠️') ? 'warning' : 'info'}`} id="speech-status">
                {speechStatus.includes('❌') && <AlertOctagon size={18} />}
                {speechStatus.includes('⚠️') && <AlertTriangle size={18} />}
                {speechStatus.includes('🎤') && <Mic size={18} />}
                {speechStatus.includes('🔇') && <MicOff size={18} />}
                {speechStatus.replace(/[❌⚠️🎤🔇]/g, '')}
              </div>
            )}

            {isRecording && (
              <div className="recording-indicator" id="recording-indicator">
                <span className="recording-dot"></span>
                Recording...
              </div>
            )}

            {/* Transcript (smaller, secondary) */}
            <div className="transcript-mini" id="transcript-mini">
              <span className="transcript-label">Transcript:</span>
              <span className={`transcript-inline ${!transcript ? 'empty' : ''}`}>
                {transcript || 'Waiting for speech...'}
              </span>
            </div>

            {/* Sign Language Display */}
            {signTokens.length > 0 && (
              <div className="sign-display-section">
                <div className="sign-display-header">
                  <h4>Sign Language Translation</h4>
                  <div className="sign-controls-inline">
                    {!isAnimating ? (
                      <button className="sign-ctrl-btn" onClick={playSignAnimation} title="Play animation">
                        <Play size={16} />
                      </button>
                    ) : (
                      <button className="sign-ctrl-btn" onClick={stopSignAnimation} title="Pause animation">
                        <Pause size={16} />
                      </button>
                    )}
                    <button className="sign-ctrl-btn" onClick={skipToNextSign} title="Next sign" disabled={activeSignIndex >= signTokens.length - 1}>
                      <SkipForward size={16} />
                    </button>
                    <div className="speed-control">
                      <label>Speed:</label>
                      <input
                        type="range"
                        min="400"
                        max="2000"
                        step="200"
                        value={2400 - animationSpeed}
                        onChange={(e) => setAnimationSpeed(2400 - Number(e.target.value))}
                      />
                    </div>
                  </div>
                </div>

                {/* Active Sign - Large Display */}
                {activeSignIndex >= 0 && activeSignIndex < signTokens.length && (
                  <div className="active-sign-display">
                    {signTokens[activeSignIndex].type === 'sign' ? (
                      <div className="active-sign-known" style={{ borderColor: signTokens[activeSignIndex].color }}>
                        <span className="active-sign-emoji">{signTokens[activeSignIndex].emoji}</span>
                        <span className="active-sign-word">{signTokens[activeSignIndex].word}</span>
                        <span className="active-sign-desc">{signTokens[activeSignIndex].desc}</span>
                        <span className="sign-type-badge known">Known Sign</span>
                      </div>
                    ) : (
                      <div className="active-sign-fingerspell">
                        <div className="fingerspell-word-label">Fingerspelling: <strong>{signTokens[activeSignIndex].word}</strong></div>
                        <div className="fingerspell-letters">
                          {signTokens[activeSignIndex].letters.map((l, li) => (
                            <div key={li} className={`fingerspell-letter ${li === activeLetterIndex ? 'active' : li < activeLetterIndex ? 'done' : ''}`}>
                              <span className="fs-hand">{l.hand}</span>
                              <span className="fs-char">{l.letter.toUpperCase()}</span>
                            </div>
                          ))}
                        </div>
                        {signTokens[activeSignIndex].letters[activeLetterIndex] && (
                          <div className="fingerspell-current-desc">
                            <strong>{signTokens[activeSignIndex].letters[activeLetterIndex].letter.toUpperCase()}</strong>: {signTokens[activeSignIndex].letters[activeLetterIndex].desc}
                          </div>
                        )}
                        <span className="sign-type-badge fingerspell">Fingerspelling</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Sign Timeline - All words */}
                <div className="sign-timeline" ref={signDisplayRef}>
                  {signTokens.map((token, idx) => (
                    <div
                      key={idx}
                      className={`sign-card ${idx === activeSignIndex ? 'active' : ''} ${idx < activeSignIndex ? 'done' : ''} ${token.type}`}
                      onClick={() => { setActiveSignIndex(idx); setActiveLetterIndex(0); }}
                    >
                      {token.type === 'sign' ? (
                        <>
                          <span className="sign-card-emoji">{token.emoji}</span>
                          <span className="sign-card-word">{token.word}</span>
                        </>
                      ) : (
                        <>
                          <span className="sign-card-emoji">🤟</span>
                          <span className="sign-card-word">{token.word}</span>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="controls">
              {!isRecording ? (
                <button className="button button-primary" onClick={startSpeechRecognition} id="btn-start-recording">
                  <Mic size={18} /> Start Recording
                </button>
              ) : (
                <button className="button button-danger" onClick={stopSpeechRecognition} id="btn-stop-recording">
                  <Square size={18} /> Stop Recording
                </button>
              )}
              <button className="button button-secondary" onClick={clearTranscript} id="btn-clear-transcript">
                <Trash2 size={18} /> Clear
              </button>
            </div>
          </div>
        </div>
      )}

      {showCopyToast && (
        <div className="copy-toast" id="copy-toast">
          <CheckCircle size={18} /> Copied to clipboard!
        </div>
      )}
    </div>
  )
}

export default Communication
