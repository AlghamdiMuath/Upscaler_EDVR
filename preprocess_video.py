import cv2
import os
import glob

def extract_frames(video_path, output_dir, frame_rate=1):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(fps // frame_rate) == 0:
            frame_filename = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()
    if count == 0:
        raise RuntimeError("No frames extracted; check the input video format.")
    print("Frame extraction complete.")

def batch_process_videos(input_folder, output_base_folder, frame_rate=1):
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in '{input_folder}'.")

    for video_path in video_files:
        video_name = os.path.basename(video_path).split('.')[0]
        output_dir = os.path.join(output_base_folder, video_name)
        try:
            extract_frames(video_path, output_dir, frame_rate=frame_rate)
        except Exception as e:
            print(f"Error processing '{video_path}': {e}")
            continue
