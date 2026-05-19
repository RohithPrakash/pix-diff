# pix-diff

Generate pixel difference visualization from video with GPU acceleration and advanced compression.

## Features

- **Pixel Difference Visualization**: Compare consecutive frames and highlight changed pixels
- **Two Modes**:
  - Grayscale: intensity based on pixel difference magnitude
  - Color: changed pixels keep their original color
- **After-Images (Motion Trails)**: Persistent fading echoes of pixel changes
- **GPU Acceleration**: CUDA support via CuPy for batch processing
- **Parallel Pipeline**: Multi-threaded producer-consumer architecture
- **Advanced Compression**: FFmpeg integration with H.264/H.265/VP9/AV1 codecs
- **Noise Threshold**: Configurable threshold to ignore minor pixel changes

## Installation

### Basic Installation

```bash
uv pip install -e .
```

### With GPU Support

```bash
uv pip install -e ".[gpu]"
```

Requires NVIDIA GPU with CUDA 12.x and CuPy.

## Usage

### Basic Usage

```bash
# Grayscale mode (default): changed pixels are white, intensity based on difference
pix-diff input.mp4

# Color mode: changed pixels keep original color, unchanged are black
pix-diff input.mp4 --mode color

# With noise threshold (ignore differences below 10)
pix-diff input.mp4 --threshold 10
```

### Advanced Options

```bash
# After-image motion trails (default: exponential fade)
pix-diff input.mp4 --after-image

# Fixed-duration linear fade (visible for 5 frames)
pix-diff input.mp4 --after-image --fade-mode fixed --fade-duration 5

# Custom exponential decay (lower = faster fade)
pix-diff input.mp4 --after-image --decay-factor 0.7

# Additive trail accumulation (trails brighten)
pix-diff input.mp4 --after-image --trail-mode additive

# GPU acceleration
pix-diff input.mp4 --gpu

# Custom batch size (default: auto-detected from VRAM)
pix-diff input.mp4 --gpu --batch-size 8

# Video compression with H.265 and quality settings
pix-diff input.mp4 --codec h265 --crf 23 --preset medium

# Lossless compression
pix-diff input.mp4 --codec h264 --crf 0

# Fast encoding for testing
pix-diff input.mp4 --preset ultrafast

# Custom output path
pix-diff input.mp4 --output result.mp4
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Visualization mode: `grayscale` or `color` | `grayscale` |
| `--threshold` | Noise threshold (0-255) | `0` |
| `--after-image` | Enable motion trail after-images | disabled |
| `--fade-mode` | Trail fade: `exponential` or `fixed` | `exponential` |
| `--fade-duration` | Frames for fixed mode fade | `10` |
| `--decay-factor` | Exponential decay factor (0.0-1.0) | `0.85` |
| `--trail-mode` | Trail accumulation: `max`, `additive`, `replace` | `max` |
| `--gpu` | Enable CUDA acceleration | disabled |
| `--batch-size` | GPU batch size | auto-detected |
| `--codec` | Output codec: `h264`, `h265`, `vp9`, `av1` | `h264` |
| `--crf` | Quality (0=lossless, 51=worst) | `23` |
| `--preset` | Speed: `ultrafast`, `fast`, `medium`, `slow`, `veryslow` | `medium` |
| `--output` | Output file path | `INPUT_diff.mp4` |

## How It Works

1. **Extract Metadata**: FPS, resolution, duration, codec via OpenCV
2. **Parallel Pipeline**: 
   - Reader thread reads frames from disk
   - Batcher groups frames into consecutive pairs
   - Compute thread processes diffs (GPU or CPU)
   - Writer thread encodes output video
3. **Pixel Comparison**: Per-channel RGB absolute difference with configurable threshold
4. **Frame Generation**: k-1 diff frames compiled into output video at original FPS

## Architecture

```
[Disk Reader] → [Batcher] → [GPU/CPU Compute] → [Video Writer]
     ↓              ↓              ↓                  ↓
  Queue A       Queue B       Queue C          [FFmpeg/OpenCV]
```

## Modes

### Grayscale Mode
- Changed pixels are white with intensity inversely proportional to difference
- Greater difference = darker (less bright)
- Unchanged pixels are black

### Color Mode
- Changed pixels keep their original color from frame 2
- Unchanged pixels are black

### After-Images (Motion Trails)

After-images create persistent, fading visual echoes of pixel changes:

- **Exponential fade** (default): Trails decay smoothly using a multiplier each frame
- **Fixed fade**: Trails fade linearly over a fixed number of frames
- **Trail accumulation modes**:
  - `max`: Keep the brightest trail value (default)
  - `additive`: Accumulate brightness from multiple changes
  - `replace`: Show only the most recent change per pixel

Trails are composited over the original diff frame using `np.maximum()`, ensuring current changes remain visible while faded trails persist in the background.

## Development

```bash
# Install with all dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=pix_diff
```

## Requirements

- Python ≥3.13
- FFmpeg (optional, for advanced codecs)
- NVIDIA GPU + CUDA 12.x (optional, for GPU acceleration)

## License

MIT
