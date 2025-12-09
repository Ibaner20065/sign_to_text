import os
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATA_FOLDER = "data"
OUTPUT_CSV = "sign_data.csv"

def extract_landmarks_from_image(image_path):
    """Extract hand landmarks from an image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    
    return None

def process_data_folder():
    """Process all images in the data folder and create CSV."""
    all_rows = []
    
    # Iterate through each gesture folder
    data_path = Path(DATA_FOLDER)
    for gesture_folder in data_path.iterdir():
        if not gesture_folder.is_dir():
            continue
        
        label = gesture_folder.name
        print(f"Processing {label}...")
        
        # Process each image in the gesture folder
        for image_file in gesture_folder.glob("*.jpg"):
            features = extract_landmarks_from_image(image_file)
            
            if features is not None:
                row = features + [label]
                all_rows.append(row)
                print(f"  ✓ {image_file.name}")
            else:
                print(f"  ✗ {image_file.name} (no hand detected)")
    
    if not all_rows:
        print("No data was extracted!")
        return
    
    # Create DataFrame and save to CSV
    col_names = []
    for i in range(21):  # 21 hand landmarks
        col_names.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    col_names.append("label")
    
    df = pd.DataFrame(all_rows, columns=col_names)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Saved {len(df)} samples to {OUTPUT_CSV}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    process_data_folder()
