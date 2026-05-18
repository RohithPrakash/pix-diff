"""Create a test video for end-to-end verification."""

import cv2
import numpy as np
from pathlib import Path


def create_test_video(output_path: str, frames: int = 30, fps: float = 10.0):
    """Create a synthetic test video with some moving pixels."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(frames):
        # Start with black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a moving white square
        x = (i * 20) % (width - 50)
        y = (i * 15) % (height - 50)
        frame[y:y+50, x:x+50] = [255, 255, 255]
        
        # Add some static noise (below threshold)
        noise = np.random.randint(0, 5, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        writer.write(frame)
    
    writer.release()
    print(f"Created test video: {output_path} ({frames} frames @ {fps}fps)")


if __name__ == "__main__":
    create_test_video("test_input.mp4")
