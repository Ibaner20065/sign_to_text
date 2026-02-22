
📘 Sign to Text — Real-Time Sign Language Recognition (ML + MediaPipe)
Welcome to Sign to Text, a real-time hand-sign recognition system that uses MediaPipe, Machine Learning, and Text-to-Speech to convert gestures into spoken words or full sentences.
This project is ideal for beginners and advanced learners exploring computer vision, gesture recognition, and AI-driven communication tools.

✨ Features
✔ Real-time hand detection
Uses MediaPipe Hands to track 21 3D landmarks at high FPS.
✔ Custom ML sign classification
Train a RandomForestClassifier on your own hand gestures (YES / NO / HELLO / etc.)
✔ Live gesture → spoken word conversion
Predicted signs are converted to speech using pyttsx3.
✔ Sentence generation
The system can build and speak multi-word sentences based on your gestures.
✔ Easy to extend
Add new gestures simply by collecting more data and retraining.

🗂 Project Structure
sign_to_speech/
│── collect_data.py           # Collect YES/NO gesture data (press Y/N)
│── train_model.py            # Train RandomForest on collected data
│── sign_to_speech_ml.py      # Real-time recognition & speech
│── sign_data.csv             # Auto-generated dataset
│── model.joblib              # Trained ML model
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
│── assets/
│     └── hello_cat.gif       # GIF used in README

📦 Installation

Install all required packages:

python -m pip install opencv-python mediapipe pyttsx3 scikit-learn pandas numpy joblib

Works with Python 3.9–3.12.

🎥 Step 1 — Collect Training Data

Run:

python collect_data.py


Controls:

Key	Action
Y	Save current frame as label yes
N	Save current frame as label no
Q	Quit

Important:

Record 50–100 samples per sign (YES & NO minimum)

Samples are appended to sign_data.csv

Ensure your hand is visible before pressing keys

🧠 Step 2 — Train the ML Model

Train a classifier on the collected dataset:

python train_model.py


This will:

Load sign_data.csv

Normalize data using StandardScaler

Train a RandomForestClassifier

Show accuracy + classification report

Save the model bundle (model.joblib)

Example output:

Loaded 200 samples
Label distribution:
yes    100
no     100
Training model...
Accuracy: 95%
Saved trained model to model.joblib

🗣 Step 3 — Real-Time Sign → Speech

Run:

python sign_to_speech_ml.py


This will:

Open your webcam

Detect your hand

Predict YES/NO

Speak the result aloud

Controls
Key	Action
Q	Quit
🖼 How It Works
1. MediaPipe Landmark Extraction

MediaPipe gives 21 hand points → each with (x, y, z).

Total features = 63 per frame.

2. Machine Learning Classification

A RandomForestClassifier learns patterns in these 63 features.

3. Prediction Stabilization

A short rolling window makes predictions stable (no flickering).

4. Text-to-Speech Output

Predicted word is spoken using pyttsx3.

📈 Adding More Signs

You can expand the vocabulary easily:

Modify collect_data.py

Add more keys like H → “hello”, T → “thankyou”

Collect new samples

Retrain:

python train_model.py


Run inference again

You can add:

HELLO

PLEASE

STOP

OK

I LOVE YOU

Custom commands (e.g., “Open Browser”)

🧩 Requirements (Python)

requirements.txt:

opencv-python
mediapipe
pyttsx3
numpy
pandas
scikit-learn
joblib


👤 Author

Indrayudh Bandyopadhyay
ECE Undergrad • AI/ML Enthusiast
GitHub: @Ibaner20065
