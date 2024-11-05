import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageEnhance
import ffmpeg

def load_frames(input_folder):
    """
    Loads all frames from the specified folder.

    Args:
        input_folder (str): Folder containing enhanced frames.

    Returns:
        list of np.array: Loaded frames as numpy arrays.
    """
    frame_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    frames = [cv2.imread(f) for f in frame_files]
    return frames

def apply_color_correction(frame, brightness=1.0, contrast=1.0):
    """
    Applies color correction to an image frame.

    Args:
        frame (np.array): The image frame.
        brightness (float): Brightness multiplier.
        contrast (float): Contrast multiplier.

    Returns:
        np.array: Color-corrected frame.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def reassemble_video(frames, output_file, fps=24):
    """
    Reassembles frames into a video file using FFmpeg.

    Args:
        frames (list of np.array): List of frames to reassemble.
        output_file (str): Path to save the output video.
        fps (int): Frames per second for the video.

    Returns:
        None
    """
    height, width, layers = frames[0].shape
    temp_folder = "./temp_frames"
    os.makedirs(temp_folder, exist_ok=True)

    # Save frames temporarily for FFmpeg
    for idx, frame in enumerate(frames):
        temp_file = os.path.join(temp_folder, f"frame_{idx:04d}.jpg")
        cv2.imwrite(temp_file, frame)

    # Use FFmpeg to reassemble the frames into a video
    (
        ffmpeg
        .input(os.path.join(temp_folder, 'frame_%04d.jpg'), framerate=fps)
        .output(output_file, pix_fmt='yuv420p')
        .run()
    )

    # Clean up temporary files
    for f in glob.glob(os.path.join(temp_folder, '*.jpg')):
        os.remove(f)
    os.rmdir(temp_folder)

# Example usage
input_folder = "./enhanced_frames"
output_file = "enhanced_video.mp4"
fps = 24

# Load frames and apply optional color correction
frames = load_frames(input_folder)
processed_frames = [apply_color_correction(frame, brightness=1.2, contrast=1.1) for frame in frames]

# Reassemble frames into a video
reassemble_video(processed_frames, output_file, fps)
print("Video reassembly complete!")
