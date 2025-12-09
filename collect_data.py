import cv2
import os
from datetime import datetime

# -------- SETTINGS --------
DATA_DIR = "data"        # folder where images will be saved
LABEL = "I"            # change to "no", "hello", etc. when collecting for that class
CAMERA_INDEX = 0         # usually 0 for built-in/webcam
# --------------------------

# Make sure data/<label>/ exists
save_dir = os.path.join(DATA_DIR, LABEL)
os.makedirs(save_dir, exist_ok=True)

print(f"[INFO] Saving images to: {save_dir}")
print("[INFO] Controls:")
print("       - Press 'c' to CAPTURE an image")
print("       - Press 'q' to QUIT")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit(1)

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    # Optional: flip the frame like a mirror
    frame = cv2.flip(frame, 1)

    # Show instructions on the frame
    text = f"Label: {LABEL} | Captured: {counter} | Press 'c' to capture, 'q' to quit"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Collect Data - Click this window, then press c/q", frame)

    # ---- KEY HANDLING ----
    key = cv2.waitKey(1) & 0xFF  # check key every frame

    if key == ord('c'):
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"{LABEL}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        counter += 1
        print(f"[SAVED] {filename}")

    elif key == ord('q'):
        print("[INFO] Quitting...")
        break
    # -----------------------

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Total images captured for label '{LABEL}': {counter}")
