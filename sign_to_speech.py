import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe and TTS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Optional: make speech a bit faster/slower if you want
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 20)

def speak(text):
    """Say text out loud (non-blocking in a tiny wrapper)."""
    print("SAY:", text)
    engine.say(text)
    engine.runAndWait()

def classify_hand(hand_landmarks):
    """
    Very simple rule-based classifier:
    - Count fingers that are 'up'
    - If almost all fingers up  -> YES
    - If no fingers up          -> NO
    Else return empty string.
    """
    landmarks = hand_landmarks.landmark

    # Finger tip and pip indices (ignore thumb for simplicity)
    finger_tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    finger_pips = [6, 10, 14, 18]   # corresponding PIP joints

    open_fingers = 0
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            open_fingers += 1

    # Simple thresholds – you can tune these
    if open_fingers >= 3:
        return "YES"
    elif open_fingers == 0:
        return "NO"
    else:
        return ""

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_label = ""
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = hands.process(rgb)

            label = ""

            if results.multi_hand_landmarks:
                # Take first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Classify gesture
                label = classify_hand(hand_landmarks)

            # If new label detected (YES/NO), speak it once
            if label and label != last_label:
                speak(label)
                last_label = label

            # Display label on screen
            if label:
                cv2.putText(frame, label, (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 3, cv2.LINE_AA)

            # Show the frame
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Sign to Speech (YES/NO demo)", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
