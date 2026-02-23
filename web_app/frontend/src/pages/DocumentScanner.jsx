import React, { useState, useRef } from 'react'

import './DocumentScanner.css'

const DocumentScanner = () => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [scanResult, setScanResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setScanResult(null)

      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.readAsDataURL(selectedFile)
    }
  }

  const handleDropZoneClick = () => {
    fileInputRef.current?.click()
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      setFile(droppedFile)
      setScanResult(null)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.readAsDataURL(droppedFile)
    }
  }

  const handleScan = async () => {
    if (!file) return

    setLoading(true)

    // Simulate scanning delay
    await new Promise((resolve) => setTimeout(resolve, 1500))

    // Mock OCR text (since backend OCR isn't available)
    const mockText = "Mock Report: Total Amount 50000. Admin Fee 5000. Drug: XyzFakeDrug. Surcharge applied. Patient: John Doe. Date: 2026-02-23."

    // Scam detection (same logic as the backend)
    const scamKeywords = [
      "admin fee", "surcharge", "processing fee", "inflated",
      "XyzFakeDrug", "hidden charge", "service tax", "convenience fee",
      "duplicate", "overcharge"
    ]
    const flaggedItems = scamKeywords.filter((word) =>
      mockText.toLowerCase().includes(word.toLowerCase())
    )

    // Check for high amounts
    const amounts = mockText.match(/\b\d{5,}\b/g) || []
    const highAmounts = amounts
      .filter((amt) => parseInt(amt) > 10000)
      .map((amt) => `High amount: ₹${amt}`)

    const allFlagged = [...flaggedItems, ...highAmounts]

    setScanResult({
      text: mockText,
      flagged: allFlagged,
      is_suspicious: allFlagged.length > 0,
    })

    setLoading(false)
  }

  const highlightText = (text, flaggedItems) => {
    if (!flaggedItems || flaggedItems.length === 0) return text

    let highlightedText = text
    flaggedItems.forEach((item) => {
      // Skip "High amount: ..." entries for regex highlighting
      if (item.startsWith('High amount:')) return
      const regex = new RegExp(`(${item.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
      highlightedText = highlightedText.replace(
        regex,
        '<mark class="flagged-text">$1</mark>'
      )
    })
    return highlightedText
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setScanResult(null)
  }

  return (
    <div id="scanner-page">
      <h1 className="page-title">📄 Document Scanner</h1>

      <div className="scanner-container">
        <div className="card">
          <h3>Upload Medical Document</h3>
          <p className="description">
            Upload a medical bill or report (JPG, PNG) to scan for suspicious content, inflated charges, and potential scam indicators.
          </p>

          <div className="upload-section">
            <div
              className={`upload-dropzone ${file ? 'has-file' : ''}`}
              onClick={handleDropZoneClick}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              id="upload-dropzone"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.pdf"
                onChange={handleFileChange}
                className="file-input"
                id="file-input"
              />
              {file ? (
                <>
                  <span className="upload-icon">✅</span>
                  <p className="upload-text"><strong>{file.name}</strong></p>
                  <p className="upload-hint">Click to change file</p>
                </>
              ) : (
                <>
                  <span className="upload-icon">📁</span>
                  <p className="upload-text">Drop your document here or click to browse</p>
                  <p className="upload-hint">Supports JPG, PNG, PDF</p>
                </>
              )}
            </div>
          </div>

          {preview && (
            <div className="preview-section" id="preview-section">
              <img src={preview} alt="Document Preview" className="preview-image" />
            </div>
          )}

          <div className="scan-actions">
            <button
              className="button button-primary"
              onClick={handleScan}
              disabled={!file || loading}
              id="scan-button"
            >
              {loading ? (
                <>
                  <span className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }}></span>
                  Scanning...
                </>
              ) : (
                '🔍 Scan Document'
              )}
            </button>
            {file && (
              <button className="button button-ghost" onClick={handleReset} id="reset-button">
                🔄 Reset
              </button>
            )}
          </div>
        </div>

        {scanResult && (
          <div className="results-container" id="scan-results">
            <div className="card">
              <h3>📝 Raw OCR Text</h3>
              <div className="ocr-text" id="ocr-text">
                {scanResult.text || 'No text extracted'}
              </div>
            </div>

            <div className="card">
              <h3>🔎 Analysis Result</h3>
              {scanResult.is_suspicious ? (
                <div>
                  <div className="flagged-badge" id="suspicious-badge">
                    ⚠️ Suspicious Content Detected
                  </div>
                  <div className="flagged-items">
                    <h4>Flagged Keywords & Issues:</h4>
                    <ul>
                      {scanResult.flagged.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="highlighted-text">
                    <h4>Text with Highlights:</h4>
                    <div
                      dangerouslySetInnerHTML={{
                        __html: highlightText(scanResult.text, scanResult.flagged),
                      }}
                    />
                  </div>
                </div>
              ) : (
                <div className="clean-badge" id="clean-badge">
                  ✅ No suspicious content detected — Document appears clean
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DocumentScanner
