"""Main pipeline orchestration for pix-diff."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .video import VideoReader
from .compression import AutoVideoWriter, detect_ffmpeg
from .diff_engine import DiffMode
from .gpu_backend import is_cuda_available
from .pipeline import VideoPipeline
from .after_image import AfterImageAccumulator, FadeMode, TrailMode
from .metrics import get_metric, get_metric_range
from .transition import TransitionController


def generate_output_path(
    input_path: str,
    mode: str = 'grayscale',
    metric: str = 'per_channel',
    threshold: int = 0,
    after_image: bool = False,
    fade_mode: str = 'exponential',
    trail_mode: str = 'max',
    decay_factor: float = 0.85,
    feather: int = 0,
    codec: str = 'h264',
    crf: int = 23,
    preset: str = 'medium',
    transition_delay: float = 0.0,
    transition_duration: float = 0.0
) -> str:
    """
    Generate default output path with descriptive filename.
    
    Format: <original>_<mode>_<metric>_t<threshold>[_ai_<fade>_<trail>_d<decay>]_f<feather>_<codec>_crf<crf>_<preset>[_tr<delay>s<duration>s].mp4
    
    Examples:
    - video_grayscale_per_channel_t0_f0_h264_crf23_medium.mp4
    - video_color_euclidean_t50_ai_exp_max_d0.85_f2_h265_crf23_medium.mp4
    - video_grayscale_per_channel_t0_f0_h264_crf23_medium_tr2s3s.mp4
    """

    path = Path(input_path)
    
    # Format threshold (perceptual metrics use decimal)
    if metric.startswith('delta_e'):
        threshold_str = f"t{threshold}.0"
    else:
        threshold_str = f"t{threshold}"
    
    # Abbreviate fade mode
    fade_abbr = 'exp' if fade_mode == 'exponential' else 'fix'
    
    # Abbreviate trail mode
    trail_abbr = {
        'max': 'max',
        'additive': 'add',
        'replace': 'rep'
    }.get(trail_mode, trail_mode)
    
    # Build filename parts
    parts = [
        path.stem,
        mode,
        metric,
        threshold_str,
    ]
    
    # Add after-image section if enabled
    if after_image:
        parts.append(f"ai_{fade_abbr}_{trail_abbr}_d{decay_factor}")
    
    # Add feather
    parts.append(f"f{feather}")
    
    # Add compression settings
    parts.append(f"{codec}_crf{crf}_{preset}")
    
    # Add transition info if active
    if transition_duration > 0:
        parts.append(f"tr{transition_delay:.0f}s{transition_duration:.0f}s")
    
    # Join and add extension
    filename = "_".join(parts) + path.suffix
    
    return str(path.parent / filename)


def process_video(input_path: str, mode: DiffMode, threshold: int = 0,
                  output_path: Optional[str] = None,
                  use_gpu: bool = False,
                  batch_size: Optional[int] = None,
                  codec: str = 'h264',
                  crf: int = 23,
                  preset: str = 'medium',
                  after_image: bool = False,
                  fade_mode: str = 'exponential',
                  fade_duration: int = 10,
                  decay_factor: float = 0.85,
                  trail_mode: str = 'max',
                  metric: str = 'per_channel',
                  feather: int = 0,
                  transition_delay: float = 0.0,
                  transition_duration: float = 0.0) -> str:
    """
    Process video and generate diff visualization.

    Args:
        input_path: Path to input video
        mode: DiffMode.GRAYSCALE or DiffMode.COLOR
        threshold: Noise threshold (0-255)
        output_path: Optional output path (auto-generated if None)
        use_gpu: Enable CUDA acceleration if available
        batch_size: GPU batch size (auto-detected if None)
        codec: Output codec (h264, h265, vp9, av1)
        crf: Compression quality (0-51, lower=better)
        preset: Encoding speed preset
        after_image: Enable after-image motion trails
        fade_mode: exponential or fixed
        fade_duration: Frames for fixed mode fade
        decay_factor: Multiplier for exponential decay
        trail_mode: max, additive, or replace
        metric: Pixel comparison metric (per_channel, euclidean, luminance,
                weighted_rgb, delta_e_cie76, delta_e_cie94)
        feather: Gaussian blur radius for smoothing edges (0=off)
        transition_delay: Seconds to show original before transition starts
        transition_duration: Duration of transition in seconds (0=disabled)

    Returns:
        Path to output video
    """
    if output_path is None:
        output_path = generate_output_path(
            input_path=input_path,
            mode=mode.value,
            metric=metric,
            threshold=threshold,
            after_image=after_image,
            fade_mode=fade_mode,
            trail_mode=trail_mode,
            decay_factor=decay_factor,
            feather=feather,
            codec=codec,
            crf=crf,
            preset=preset,
            transition_delay=transition_delay,
            transition_duration=transition_duration
        )
    
    # Check GPU availability
    if use_gpu and not is_cuda_available():
        print("Warning: CUDA not available, falling back to CPU processing")
        use_gpu = False
    
    with VideoReader(input_path) as reader:
        meta = reader.metadata
        print(f"Input: {input_path}")
        print(f"  {meta}")
        print(f"  Mode: {mode.value}, Threshold: {threshold}")
        print(f"  Metric: {metric}")
        print(f"  Feather: {feather}px" if feather > 0 else "  Feather: off")
        if transition_duration > 0:
            print(f"  Transition: delay={transition_delay}s, duration={transition_duration}s")
        print(f"  GPU: {'enabled' if use_gpu else 'disabled'}")
        print(f"  After-images: {'enabled' if after_image else 'disabled'}")
        
        # Get metric function
        metric_fn = get_metric(metric)
        metric_range = get_metric_range(metric)
        
        # Setup after-image accumulator if enabled
        accumulator = None
        if after_image:
            accumulator = AfterImageAccumulator(
                frame_shape=(meta.height, meta.width, 3),
                fade_mode=FadeMode(fade_mode),
                fade_duration=fade_duration,
                decay_factor=decay_factor,
                trail_mode=TrailMode(trail_mode)
            )
            print(f"    Fade: {fade_mode}, duration={fade_duration}, "
                  f"decay={decay_factor}, trail={trail_mode}")
        
        frame_count = meta.total_frames
        if frame_count < 2:
            raise ValueError("Video must have at least 2 frames")
        
        # Setup transition controller if enabled
        transition = None
        if transition_duration > 0:
            transition = TransitionController(
                fps=meta.fps,
                delay_sec=transition_delay,
                duration_sec=transition_duration
            )
            print(f"  {transition}")
        
        # Setup writer with compression
        with AutoVideoWriter(
            output_path, meta.fps, meta.width, meta.height,
            codec=codec, crf=crf, preset=preset
        ) as writer:
            
            # Create and run pipeline
            pipeline = VideoPipeline(
                reader=reader,
                writer=writer,
                mode=mode,
                threshold=threshold,
                batch_size=batch_size,
                use_gpu=use_gpu,
                after_image=accumulator,
                metric_fn=metric_fn,
                metric_name=metric,
                feather=feather,
                transition=transition
            )
            
            processed = pipeline.run()
            
            print(f"Output: {output_path}")
            print(f"  Generated {processed} diff frames")
        
        # Copy audio from original if FFmpeg available
        if detect_ffmpeg():
            _copy_audio(input_path, output_path)
    
    return output_path


def _copy_audio(input_path: str, output_path: str):
    """
    Copy audio track from input video to output using FFmpeg.
    
    Args:
        input_path: Source video with audio
        output_path: Output video (will be overwritten with audio)
    """
    import shutil
    import tempfile
    import os
    
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print("  Warning: FFmpeg not found, audio not copied")
        return
    
    try:
        # Create temporary file for output with audio
        temp_fd, temp_path = tempfile.mkstemp(suffix=Path(output_path).suffix)
        os.close(temp_fd)
        
        # Use FFmpeg to copy audio from input to output
        cmd = [
            ffmpeg_path,
            '-y',
            '-i', output_path,      # Video (no audio)
            '-i', input_path,       # Original (for audio)
            '-c:v', 'copy',         # Copy video without re-encoding
            '-c:a', 'copy',         # Copy audio without re-encoding
            '-map', '0:v:0',        # Use video from output
            '-map', '1:a:0?',       # Use audio from input (if exists)
            '-shortest',            # Match shortest stream
            temp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Replace original output with temp file
            shutil.move(temp_path, output_path)
            print("  Audio copied successfully")
        else:
            print(f"  Warning: Audio copy failed: {result.stderr[:200]}")
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"  Warning: Could not copy audio: {e}")


def main():
    """Entry point for CLI."""
    from .cli import parse_args
    
    args = parse_args()
    
    try:
        mode = DiffMode(args.mode)
        process_video(
            input_path=args.input,
            mode=mode,
            threshold=args.threshold,
            output_path=args.output,
            use_gpu=args.gpu,
            batch_size=args.batch_size,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            after_image=args.after_image,
            fade_mode=args.fade_mode,
            fade_duration=args.fade_duration,
            decay_factor=args.decay_factor,
            trail_mode=args.trail_mode,
            metric=args.metric,
            feather=args.feather,
            transition_delay=args.transition_delay,
            transition_duration=args.transition
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
