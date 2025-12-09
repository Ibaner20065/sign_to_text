import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
from collections import Counter, deque

MODEL_FILE = "sign_model.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features

def count_fingers(hand_landmarks):
    """Count how many fingers are extended."""
    landmarks = hand_landmarks.landmark
    
    # Finger tips and PIPs (indices, middle, ring, pinky - skip thumb)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    open_fingers = 0
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            open_fingers += 1
    
    return open_fingers

def classify_by_fingers(open_fingers):
    """Classify gesture based on finger count."""
    if open_fingers >= 3:
        return "YES", 0.95  # High confidence for many fingers
    elif open_fingers == 2:
        return "hello", 0.95  # 2 fingers up = hello
    elif open_fingers == 1:
        return "hihi", 0.95  # 1 finger up = hihi
    elif open_fingers == 0:
        return "NO", 0.95   # High confidence for closed fist
    else:
        return None, 0.0    # Uncertain, defer to ML

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    # Load the trained model
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"Error: Could not load model '{MODEL_FILE}'. Did you run train_model.py?")
        print(e)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sentence_words = []
    last_word = None

    # Keep last N predictions to stabilize
    PREDICTION_WINDOW = 12
    preds_buffer = deque(maxlen=PREDICTION_WINDOW)

    print("Instructions:")
    print(" - Show a sign in front of the camera.")
    print(" - Model predicts a word from your trained labels.")
    print(" - When a word is stable, it gets added to the sentence.")
    print(" - Press 's' to SPEAK the current sentence.")
    print(" - Press 'c' to CLEAR the sentence.")
    print(" - Press 'q' to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_prediction = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                try:
                    features = extract_hand_features(hand_landmarks)
                    features = np.array(features).reshape(1, -1)

                    # Predict label using ML model
                    current_prediction = model.predict(features)[0]
                    
                    # Get prediction confidence from ML model
                    proba = model.predict_proba(features)[0]
                    ml_confidence = max(proba) * 100
                    
                    # If ML confidence is low, use finger counting
                    if ml_confidence < 70:
                        open_fingers = count_fingers(hand_landmarks)
                        finger_pred, finger_conf = classify_by_fingers(open_fingers)
                        
                        if finger_pred is not None:
                            # Use finger counting result
                            current_prediction = finger_pred
                            confidence = finger_conf * 100
                            print(f"  [Finger count: {open_fingers} → {finger_pred}]")
                        else:
                            # Keep ML prediction even with low confidence
                            confidence = ml_confidence
                    else:
                        # Use ML prediction (high confidence)
                        confidence = ml_confidence
                    
                    preds_buffer.append(current_prediction)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    preds_buffer.append("")
                    confidence = 0
            else:
                preds_buffer.append("")

            # Use majority vote over last N frames
            if len(preds_buffer) == PREDICTION_WINDOW:
                # Ignore empty entries
                non_empty = [p for p in preds_buffer if p != ""]
                if non_empty:
                    most_common, count = Counter(non_empty).most_common(1)[0]
                    # Basic confidence threshold (at least 60% of window)
                    if count >= int(0.6 * PREDICTION_WINDOW):
                        if most_common != last_word:
                            last_word = most_common
                            sentence_words.append(most_common)
                            print(f"Added word: {most_common}")
                            preds_buffer.clear()

            # Draw sentence and current word on screen
            sentence_text = " ".join(sentence_words)
            cv2.putText(frame, f"Sentence: {sentence_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            if current_prediction:
                cv2.putText(frame, f"Live: {current_prediction} ({confidence:.0f}%)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

            cv2.putText(frame, "q: quit | s: speak | c: clear",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            cv2.imshow("Sign to Sentence (ML)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                if sentence_words:
                    sentence_text = " ".join(sentence_words)
                    print(f"Speaking: {sentence_text}")
                    speak(sentence_text)
                else:
                    print("Sentence is empty.")
            elif key == ord('c'):
                sentence_words = []
                last_word = None
                preds_buffer.clear()
                print("Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
