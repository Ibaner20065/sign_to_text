import React, { useState, useRef, useEffect, useCallback } from 'react'
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

const Communication = () => {
  const [mode, setMode] = useState('sign')
  const [sentence, setSentence] = useState('')
  const [currentWord, setCurrentWord] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [showCopyToast, setShowCopyToast] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)
  const predictionBufferRef = useRef([])
  const lastWordRef = useRef(null)
  const recognitionRef = useRef(null)

  const PREDICTION_WINDOW = 12
  const CONFIDENCE_THRESHOLD = 0.6

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

  const extractHandFeatures = (landmarks) => {
    const features = []
    landmarks.forEach((lm) => {
      features.push(lm.x, lm.y, lm.z)
    })
    return features
  }

  const onResults = async (results) => {
    const canvasCtx = canvasRef.current.getContext('2d')
    canvasCtx.save()
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const handLandmarks = results.multiHandLandmarks[0]
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
        const features = extractHandFeatures(handLandmarks)

        // Mock prediction (no backend needed)
        const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        const index = Math.floor(features[0] * 26) % 26
        const prediction = letters[index] || 'A'
        const confidence = 0.85 + Math.random() * 0.1

        if (prediction) {
          setCurrentWord(prediction)
          predictionBufferRef.current.push(prediction)

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
          }
        }
      } catch (error) {
        console.error('Prediction error:', error)
      }
    } else {
      setCurrentWord('')
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
    predictionBufferRef.current = []
    lastWordRef.current = null
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(sentence)
    setShowCopyToast(true)
    setTimeout(() => setShowCopyToast(false), 2000)
  }

  const finalTranscriptRef = useRef('')

  const startSpeechRecognition = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition not supported in this browser')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    finalTranscriptRef.current = transcript

    recognition.onresult = (event) => {
      let interimTranscript = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscriptRef.current += t + ' '
        } else {
          interimTranscript += t
        }
      }

      setTranscript(finalTranscriptRef.current + interimTranscript)
    }

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      setIsRecording(false)
    }

    recognition.onend = () => {
      setIsRecording(false)
    }

    recognition.start()
    recognitionRef.current = recognition
    setIsRecording(true)
  }

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
      setIsRecording(false)
    }
  }

  const clearTranscript = () => {
    setTranscript('')
  }

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
            <div className="prediction-display">
              <div className="current-word" id="current-word">
                Current: <strong>{currentWord || 'No sign detected'}</strong>
              </div>
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
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="speech-mode">
          <div className="card">
            <h3>Live Captioning</h3>
            <p className="description">
              Speak into your microphone. Your words will be transcribed in real-time so a deaf user can read along.
            </p>

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
