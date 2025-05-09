import os
import glob
import json

# ALIGN_DIR = 'Dataset/alignments/train'
ALIGN_DIR = 'Dataset/alignments/test'
# VIDEO_DIR = 'Dataset/videos/train'
VIDEO_DIR = 'Dataset/videos/test'
OUTPUT_FILE = 'data/test_segments.json'
FRAME_RATE = 1000 / 40  # 25 fps -> 1 frame = 40 ms

def parse_align_file(align_path, video_filename):
    segments = []
    with open(align_path, 'r') as f:
        for line in f:
            start, end, word = line.strip().split()
            if word == 'sil':
                continue  # skip silence
            start_frame = int(int(start) / FRAME_RATE)
            end_frame = int(int(end) / FRAME_RATE)
            segments.append({
                "video": video_filename.lower(),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "label": word.lower()
            })
    return segments

def main():
    all_segments = []
    for align_file in glob.glob(os.path.join(ALIGN_DIR, '*.align')):
        base_name = os.path.splitext(os.path.basename(align_file))[0]
        video_path = os.path.join(VIDEO_DIR, f'{base_name}.mpg')
        if not os.path.exists(video_path):
            print(f'[WARN] Video not found for: {base_name}')
            continue
        segments = parse_align_file(align_file, video_path)
        all_segments.extend(segments)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_segments, f, indent=2)
    print(f"Saved {len(all_segments)} segments to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
