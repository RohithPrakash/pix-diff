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

# With noise threshold (ignore differences below threshold)
# Scale depends on metric: 0-255 for most, perceptual units for delta_e_* metrics
pix-diff input.mp4 --threshold 10
pix-diff input.mp4 --metric delta_e_cie76 --threshold 5.0
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

### Edge Smoothing (Feathering)

```bash
# Default sharp edges
pix-diff input.mp4

# Soft 2-pixel feathered edges (removes jagged artifacts)
pix-diff input.mp4 --feather 2

# Strong smoothing for artistic effect
pix-diff input.mp4 --feather 5 --after-image

# Combine with color mode for soft glow effect
pix-diff input.mp4 --feather 3 --mode color
```

### Transition Effect

```bash
# Smooth 3-second transition starting immediately
pix-diff input.mp4 --transition 3

# Show original for 2 seconds, then transition over 5 seconds
pix-diff input.mp4 --transition 5 --transition-delay 2

# Combine with after-images for smooth intro
pix-diff input.mp4 --transition 3 --transition-delay 1 --after-image
```

### Pixel Comparison Metrics

```bash
# Per-channel metric (default): independent RGB differences
pix-diff input.mp4

# Euclidean distance: 3D spatial color difference
pix-diff input.mp4 --metric euclidean --threshold 50

# Luminance only: brightness changes (Rec.709)
pix-diff input.mp4 --metric luminance --threshold 20

# Weighted RGB: luminance-weighted differences
pix-diff input.mp4 --metric weighted_rgb --threshold 15

# CIE76 Delta E: perceptual color difference
pix-diff input.mp4 --metric delta_e_cie76 --threshold 5.0

# CIE94 Delta E: improved perceptual metric
pix-diff input.mp4 --metric delta_e_cie94 --threshold 2.0
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Visualization mode: `grayscale` or `color` | `grayscale` |
| `--threshold` | Noise threshold. Scale depends on metric: 0-255 for most, perceptual units for `delta_e_*` | `0` |
| `--metric` | Pixel comparison metric: `per_channel`, `euclidean`, `luminance`, `weighted_rgb`, `delta_e_cie76`, `delta_e_cie94` | `per_channel` |
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
| `--feather` | Gaussian blur radius for smoothing edges (0=off) | `0` |
| `--transition` | Transition duration from original to diff in seconds (0=disabled) | `0` |
| `--transition-delay` | Seconds to show original before transition starts | `0` |
| `--output` | Output file path (auto-generated if omitted) | auto-generated |

## How It Works

1. **Extract Metadata**: FPS, resolution, duration, codec via OpenCV
2. **Parallel Pipeline**: 
   - Reader thread reads frames from disk
   - Batcher groups frames into consecutive pairs
   - Compute thread processes diffs (GPU or CPU)
   - Writer thread encodes output video
3. **Pixel Comparison**: Configurable metric (per-channel, Euclidean, luminance, perceptual) with configurable threshold
4. **Frame Generation**: k-1 diff frames compiled into output video at original FPS

## Architecture

```
[Disk Reader] → [Batcher] → [GPU/CPU Compute] → [Video Writer]
     ↓              ↓              ↓                  ↓
  Queue A       Queue B       Queue C          [FFmpeg/OpenCV]
```

## Output Naming Convention

When `--output` is not specified, the output filename is auto-generated based on all settings used:

```
<original>_<mode>_<metric>_t<threshold>[_ai_<fade>_<trail>_d<decay>]_f<feather>_<codec>_crf<crf>_<preset>[_tr<delay>s<duration>s].mp4
```

**Examples:**

| Command | Output Filename |
|---------|----------------|
| `pix-diff video.mp4` | `video_grayscale_per_channel_t0_f0_h264_crf23_medium.mp4` |
| `pix-diff video.mp4 --mode color --metric euclidean --threshold 50` | `video_color_euclidean_t50_f0_h264_crf23_medium.mp4` |
| `pix-diff video.mp4 --after-image --feather 2 --codec h265` | `video_grayscale_per_channel_t0_ai_exp_max_d0.85_f2_h265_crf23_medium.mp4` |
| `pix-diff video.mp4 --metric delta_e_cie76 --threshold 5 --after-image --fade-mode fixed --trail-mode additive --decay-factor 0.7 --feather 3 --codec h265 --crf 20 --preset fast` | `video_grayscale_delta_e_cie76_t5.0_ai_fix_add_d0.7_f3_h265_crf20_fast.mp4` |
| `pix-diff video.mp4 --transition 3 --transition-delay 2` | `video_grayscale_per_channel_t0_f0_h264_crf23_medium_tr2s3s.mp4` |

**Naming components:**
- **mode**: `grayscale` or `color`
- **metric**: metric name (e.g., `per_channel`, `delta_e_cie76`)
- **threshold**: `t<int>` (or `t<float>.0` for perceptual metrics)
- **after-image** (only if enabled): `ai_<fade>_<trail>_d<decay>`
  - fade: `exp` (exponential) or `fix` (fixed)
  - trail: `max`, `add` (additive), or `rep` (replace)
- **feather**: `f<int>` (0 = off)
- **compression**: `<codec>_crf<crf>_<preset>`
- **transition** (only if enabled): `tr<delay>s<duration>s`

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

After-images are fully compatible with GPU batch processing — diff frames are computed in batches while trails are applied sequentially, preserving performance and visual correctness.

## Metrics

### Per-Channel (Default)
- Independent RGB channel absolute differences
- Range: 0-255
- Fastest, suitable for most use cases

### Euclidean
- 3D Euclidean distance in RGB color space
- Range: 0-441.67
- Accounts for combined channel differences

### Luminance (Rec.709)
- Brightness-only comparison using Rec.709 coefficients
- Range: 0-255
- Ignores chroma (color) changes

### Weighted RGB
- Luminance-weighted RGB differences
- Range: 0-255
- Balances color and brightness sensitivity

### Delta E CIE76
- Euclidean distance in CIELAB perceptual color space
- Range: 0-100+ (perceptual units)
- Human vision approximated

### Delta E CIE94
- Improved perceptual difference with parametric weights
- Range: 0-100+ (perceptual units)
- Industry standard for color accuracy

**Note**: Perceptual metrics (`delta_e_*`) run on CPU. Other metrics support GPU acceleration.

## Edge Smoothing (Feathering)

By default, changed pixels have sharp binary edges (either fully on or fully off). Feathering applies a Gaussian blur to the change mask, creating smooth transitions at boundaries:

- **Radius 0** (default): Sharp binary edges
- **Radius 1-2**: Subtle softening, removes pixelation artifacts
- **Radius 3-5**: Strong smoothing, artistic glow effect
- **Radius 5+**: Very soft, dreamlike transitions

Feathering works with all modes (grayscale, color) and combines well with after-images for smooth motion trails.

## Transition Effect

The transition effect creates a smooth crossfade from the original video to the pixel difference visualization:

```
[Original] → [Transition] → [Diff]
     ↓             ↓           ↓
  Alpha=0    Alpha 0→1     Alpha=1
```

- **Original phase** (`--transition-delay`): Show original video footage
- **Transition phase** (`--transition`): Smooth ease-in-out blend from original to diff
- **Diff phase**: Show pure pixel difference visualization

**Formula**: `output = original × (1-α) + diff × α`

Where α follows a smoothstep curve for natural acceleration/deceleration.

**Features:**
- Original audio is preserved and copied to output
- Works with all modes, metrics, and effects
- After-image trails fade in naturally as α increases
- Configurable delay and duration

**Examples:**

```bash
# 3-second smooth transition starting immediately
pix-diff input.mp4 --transition 3

# Show original for 2 seconds, then transition over 5 seconds
pix-diff input.mp4 --transition 5 --transition-delay 2

# Combine with after-images for smooth intro
pix-diff input.mp4 --transition 3 --after-image --feather 2
```

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

MIT - see [LICENSE](LICENSE).

Third-party dependencies keep their own licenses: NumPy (BSD-3-Clause),
opencv-python (Apache-2.0), tqdm (MPL-2.0 AND MIT), CuPy (MIT).
