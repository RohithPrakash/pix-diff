"""Main pipeline orchestration for pix-diff."""

import sys
from pathlib import Path
from typing import Optional

from .video import VideoReader
from .compression import AutoVideoWriter
from .diff_engine import DiffMode
from .gpu_backend import is_cuda_available
from .pipeline import VideoPipeline


def generate_output_path(input_path: str) -> str:
    """Generate default output path: input_diff.mp4"""
    path = Path(input_path)
    return str(path.parent / f"{path.stem}_diff{path.suffix}")


def process_video(input_path: str, mode: DiffMode, threshold: int = 0,
                  output_path: Optional[str] = None,
                  use_gpu: bool = False,
                  batch_size: Optional[int] = None,
                  codec: str = 'h264',
                  crf: int = 23,
                  preset: str = 'medium') -> str:
    """
    Process video and generate diff visualization.
    
    Args:
        input_path: Path to input video
        mode: DiffMode.GRAYSCALE or DiffMode.COLOR
        threshold: Noise threshold (0-255)
        output_path: Optional output path (default: input_diff.mp4)
        use_gpu: Enable CUDA acceleration if available
        batch_size: GPU batch size (auto-detected if None)
        codec: Output codec (h264, h265, vp9, av1)
        crf: Compression quality (0-51, lower=better)
        preset: Encoding speed preset
    
    Returns:
        Path to output video
    """
    if output_path is None:
        output_path = generate_output_path(input_path)
    
    # Check GPU availability
    if use_gpu and not is_cuda_available():
        print("Warning: CUDA not available, falling back to CPU processing")
        use_gpu = False
    
    with VideoReader(input_path) as reader:
        meta = reader.metadata
        print(f"Input: {input_path}")
        print(f"  {meta}")
        print(f"  Mode: {mode.value}, Threshold: {threshold}")
        print(f"  GPU: {'enabled' if use_gpu else 'disabled'}")
        
        frame_count = meta.total_frames
        if frame_count < 2:
            raise ValueError("Video must have at least 2 frames")
        
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
                use_gpu=use_gpu
            )
            
            processed = pipeline.run()
            
            print(f"Output: {output_path}")
            print(f"  Generated {processed} diff frames")
    
    return output_path


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
            preset=args.preset
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
