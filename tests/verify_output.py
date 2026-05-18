"""Verify output video properties."""

import cv2
from pathlib import Path


def check_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"{path}: {frames} frames, {fps}fps, {width}x{height}")


check_video("test_input.mp4")
check_video("test_input_diff.mp4")
check_video("test_color_diff.mp4")
