import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm

# Modify these paths as needed
VIDEO_DIR = "Dataset/videos/test"
ALIGN_DIR = "Dataset/alignments/test"
OUTPUT_DIR = "data/mouth_rois/test_rois"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

# Initialize face and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL)

def extract_mouth_roi(frame, shape):
    mouth = shape[48:68]
    x_min = np.min(mouth[:, 0])
    y_min = np.min(mouth[:, 1])
    x_max = np.max(mouth[:, 0])
    y_max = np.max(mouth[:, 1])
    
    x_margin = int((x_max - x_min) * 0.2)
    y_margin = int((y_max - y_min) * 0.2)
    
    x_min = max(x_min - x_margin, 0)
    y_min = max(y_min - y_margin, 0)
    x_max = min(x_max + x_margin, frame.shape[1])
    y_max = min(y_max + y_margin, frame.shape[0])
    
    mouth_img = frame[y_min:y_max, x_min:x_max]
    resized = cv2.resize(mouth_img, (96, 96))
    return resized

def process_video(video_path, align_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(align_path, 'r') as f:
        lines = f.readlines()

    frame_data = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_data.append(gray)

    for idx, line in enumerate(lines):
        start, end, word = line.strip().split()
        if word == "sil":  # Skip silence
            continue

        start_frame = int(int(start) / 1000 * fps / 100)
        end_frame = int(int(end) / 1000 * fps / 100)
        mouth_frames = []

        for i in range(start_frame, min(end_frame, len(frame_data))):
            gray = frame_data[i]
            dets = detector(gray)
            if len(dets) == 0:
                continue
            shape = predictor(gray, dets[0])
            coords = np.array([[pt.x, pt.y] for pt in shape.parts()])
            mouth_roi = extract_mouth_roi(gray, coords)
            mouth_frames.append(mouth_roi)

        if len(mouth_frames) > 0:
            mouth_frames = np.stack(mouth_frames)
            word_clean = word.lower()
            out_path = os.path.join(output_dir, f"{video_name}_{word_clean}_{idx:03d}.npy")
            np.save(out_path, mouth_frames)
            print(f"[SAVED] {out_path} shape: {mouth_frames.shape}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mpg")]

    for vid in tqdm(video_files):
        video_path = os.path.join(VIDEO_DIR, vid)
        align_path = os.path.join(ALIGN_DIR, vid.replace(".mpg", ".align"))
        if os.path.exists(align_path):
            process_video(video_path, align_path, OUTPUT_DIR)

if __name__ == "__main__":
    main()
