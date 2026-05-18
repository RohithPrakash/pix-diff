"""Command line interface for pix-diff."""

import argparse
from pathlib import Path


def parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate pixel difference visualization from video."
    )
    
    parser.add_argument(
        "input",
        help="Input video file path"
    )
    
    parser.add_argument(
        "--mode",
        choices=["grayscale", "color"],
        default="grayscale",
        help="Visualization mode: grayscale (white intensity = difference) or color (keep original color). Default: grayscale"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        metavar="T",
        help="Noise threshold (0-255). Pixels with difference below this are considered unchanged. Default: 0"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output video path. Default: INPUT_diff.mp4"
    )
    
    # GPU acceleration
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable CUDA GPU acceleration (requires NVIDIA GPU and CuPy)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        metavar="N",
        help="GPU batch size (default: auto-detected from VRAM)"
    )
    
    # Video compression
    parser.add_argument(
        "--codec",
        choices=["h264", "h265", "vp9", "av1"],
        default="h264",
        help="Output video codec. Default: h264"
    )
    
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        metavar="Q",
        help="Compression quality: 0=lossless, 23=default, 51=worst. Default: 23"
    )
    
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        default="medium",
        help="Encoding speed/quality tradeoff. Default: medium"
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 255:
        parser.error("Threshold must be between 0 and 255")
    
    # Validate CRF
    if not 0 <= args.crf <= 51:
        parser.error("CRF must be between 0 and 51")
    
    # Validate batch size
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("Batch size must be >= 1")
    
    # Validate input exists
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    
    return args
