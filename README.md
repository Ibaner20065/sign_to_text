# Inclusive Healthcare Platform (MVP)

An accessibility-first healthcare web platform designed for both the general public and differently-abled users. This MVP builds on an existing FastAPI backend and sign-language ML pipeline to deliver real-time communication tools, emergency awareness, document safety checks, and accessible healthcare discovery -- all behind secure authentication.

---

## 🌍 Platform Vision

Break down communication barriers in healthcare by providing inclusive, assistive interactions through browser-native capabilities and lightweight ML -- no specialized hardware or paid APIs required.

**Primary users:**

- Deaf / hard-of-hearing users
- Speech-impaired users
- General patients needing accessible tools
- Caregivers and healthcare staff

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | FastAPI (extended existing backend) |
| ML Model | RandomForest (`sign_model.pkl`) + MediaPipe Hands (browser) |
| Maps | Leaflet.js |
| Speech-to-Text | Browser Web Speech API |
| Text-to-Speech | Browser Web Speech API |
| OCR | pytesseract (existing backend) |
| Auth | JWT via FastAPI |

> No paid external APIs required.

---

## 🔐 Authentication

All application pages require login.

**Features:**
- Register (name, email, password)
- Login
- JWT stored in browser
- JWT sent as `Authorization: Bearer <token>` to FastAPI

**Backend endpoints:**
- `POST /auth/register`
- `POST /auth/login`

---

## 🧭 Navigation

The platform contains 5 authenticated pages:

1. Communication
2. Ambulance / SOS
3. Document Scanner
4. Hospital Finder
5. Profile / Settings

---

## 🤟 1. Communication -- Sign Language + Speech

A single screen with two user-toggle modes.

### Mode A -- Sign to Speech

Enable speech-impaired users to communicate verbally via sign language.

**Flow:**
1. Webcam feed captured in browser
2. MediaPipe Hands extracts landmarks
3. Landmarks sent to `POST /predict`
4. Backend returns recognized word
5. Words buffered with majority-vote smoothing
6. Sentence displayed on screen
7. Browser Text-to-Speech speaks sentence

**Controls:** Speak / Clear / Copy sentence

---

### Mode B -- Speech to Text (Live Captioning)

Allow hearing individuals to speak so that deaf users can read along in real time.

**Flow:**
1. Microphone activated via Web Speech API
2. Speech transcribed in real time
3. Text displayed as scrolling captions

---

## 🚑 2. Ambulance Tracker / Emergency SOS

An interactive emergency awareness map built with Leaflet.

**Map features:**
- Centered on browser geolocation
- 3 mock ambulance units sourced from `/ambulance-status`
- Status indicators: Moving / Stationary
- ETA and ambulance type per unit
- Animated markers simulating vehicle motion

**SOS Button:**
- Large, prominent red emergency button
- Confirmation modal before triggering
- Displays "Emergency alert sent" on confirm
- Mock response only -- no live dispatch integration

---

## 📄 3. Document Scanner -- Medical Scam Detection

Upload a medical bill or report (image or PDF) to check for suspicious content.

**Flow:**
1. File uploaded via browser
2. Sent to `POST /scan-document`
3. pytesseract extracts text from the file
4. Text scanned for suspicious patterns:
   - Inflated pricing indicators
   - Unrecognized drug names
   - Duplicate line items
5. Results displayed in two panels:
   - Raw OCR text
   - Flagged items with warning highlights

If no issues are found, the result displays: `✅ No suspicious content detected`

---

## 🏥 4. Hospital / Doctor Finder

Discover nearby healthcare providers using an accessible, map-based UI and mock spatial data.

**Features:**
- Leaflet map centered on user location
- 10-15 mock hospitals generated around user coordinates
- Pin popups display: hospital name, specialty, distance, and phone number

**Filter bar options:**
- General
- Cardiology
- Neurology
- Pediatrics
- Emergency

> No external hospital API required.

---

## 👤 5. User Profile / Accessibility Settings

Manage your account and personalize your accessibility experience.

**Profile:**
- Name and email
- Profile picture upload

**Accessibility settings:**
- Font size: Normal / Large / Extra Large
- High-contrast mode toggle
- TTS speed slider

**Account actions:**
- Change password
- Logout

---

## 🗂️ Project Structure

```
web_app/
  frontend/            # React + Vite app
    src/
      pages/           # Communication, Ambulance, Scanner, Hospitals, Profile
      components/      # Navbar, Map, WebcamFeed, SpeechPanel, etc.
      context/         # AuthContext (JWT handling)
  backend/
    main.py            # Existing FastAPI app (extended with auth + hospitals)
```

---

## 🔌 Backend Endpoints

| Category | Method | Endpoint |
|---|---|---|
| Auth | POST | `/auth/register` |
| Auth | POST | `/auth/login` |
| ML | POST | `/predict` |
| Emergency | GET | `/ambulance-status` |
| OCR | POST | `/scan-document` |
| Hospitals | GET | `/hospitals` |

---

## ♿ Accessibility Principles

This platform is built accessibility-first. Core principles:

- Large touch targets throughout
- High-contrast mode available
- Text alternatives for all audio content
- Audio alternatives for all text content
- Adjustable font sizes
- Caption-first communication design

---

## 🚀 MVP Scope

This MVP is intentionally scoped to be fully buildable using only:

- Existing FastAPI backend
- Existing sign language model (`sign_model.pkl`)
- Browser-native APIs (Web Speech, MediaPipe, Geolocation)
- Mock spatial data for hospitals and ambulances

No real-time GPS dispatch, live hospital APIs, or paid third-party services are required.

---

## 📌 Status

Architecture defined and build-ready. Frontend and backend integration in progress.

---

## 🧑‍⚕️ Goal

Build a practical, inclusive healthcare interface where communication barriers are reduced and essential healthcare interactions become accessible to every user -- regardless of ability.
