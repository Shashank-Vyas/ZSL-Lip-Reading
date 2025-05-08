import cv2
import dlib
import json
import os
import numpy as np
from tqdm import tqdm

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SEGMENT_FILE = "data/train_segments.json"
OUTPUT_DIR = "data/mouth_rois/"
IMG_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def extract_mouth(frame):
    faces = detector(frame)
    if len(faces) == 0:
        return None
    shape = predictor(frame, faces[0])
    mouth_points = np.array([[p.x, p.y] for p in shape.parts()[48:]])
    x, y, w, h = cv2.boundingRect(mouth_points)
    roi = frame[y:y+h, x:x+w]
    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

def extract_segment_frames(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth = extract_mouth(gray)
        if mouth is not None:
            frames.append(mouth)
    cap.release()
    return np.array(frames)

def main():
    with open(SEGMENT_FILE, 'r') as f:
        segments = json.load(f)

    for i, seg in tqdm(enumerate(segments), total=len(segments)):
        frames = extract_segment_frames(seg["video"], seg["start_frame"], seg["end_frame"])
        if frames.shape[0] == 0:
            continue
        video_name = os.path.splitext(os.path.basename(seg["video"]))[0].lower()
        save_path = os.path.join(OUTPUT_DIR, f"{video_name}_{i}_{seg['label']}.npy")
        np.save(save_path, frames)

    print(f"[DONE] Saved mouth ROIs to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
