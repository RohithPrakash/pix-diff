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
        "--metric",
        choices=["per_channel", "euclidean", "luminance", "weighted_rgb", "delta_e_cie76", "delta_e_cie94"],
        default="per_channel",
        help="Pixel comparison metric. Default: per_channel"
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
    
    # After-images (motion trails)
    parser.add_argument(
        "--after-image",
        action="store_true",
        help="Enable after-image motion trails. Disabled by default."
    )
    
    parser.add_argument(
        "--fade-mode",
        choices=["exponential", "fixed"],
        default="exponential",
        help="Trail fade mode: exponential (multiplicative decay) or fixed (linear fade). Default: exponential"
    )
    
    parser.add_argument(
        "--fade-duration",
        type=int,
        default=10,
        metavar="N",
        help="Frames for fixed mode fade. Default: 10"
    )
    
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.85,
        metavar="F",
        help="Exponential decay factor (0.0-1.0). Default: 0.85"
    )
    
    parser.add_argument(
        "--trail-mode",
        choices=["max", "additive", "replace"],
        default="max",
        help="Trail accumulation: max (brightest), additive (cumulative), replace (latest). Default: max"
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
    
    # Edge smoothing
    parser.add_argument(
        "--feather",
        type=int,
        default=0,
        metavar="R",
        help="Gaussian blur radius for smoothing diff edges (0=off). Higher values create softer transitions. Default: 0"
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
    
    # Validate after-image arguments
    if args.fade_duration < 1:
        parser.error("Fade duration must be >= 1")
    
    if not 0.0 <= args.decay_factor <= 1.0:
        parser.error("Decay factor must be between 0.0 and 1.0")
    
    # Validate feather
    if args.feather < 0:
        parser.error("Feather radius must be >= 0")
    
    # Validate input exists
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    
    return args
