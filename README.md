# Mandelbrot Deep Zoom Video Generator

**Generates stunning 1080p videos zooming deep into the Mandelbrot set with psychedelic color cycling.**

Features high-performance rendering optimized with Numba JIT compilation and configurable zoom targets including the famous "Seahorse Valley" region.

![Mandelbrot Set](https://upload.wikimedia.org/wikipedia/commons/2/21/Mandel_zoom_00_mandelbrot_set.jpg)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate full 60-second video (1800 frames)
python main.py

# Quick 10-frame preview
python main.py --preview

# Custom zoom target and settings
python main.py --center-real -0.8 --center-imag 0.156 --duration 30
```

## Features

- **High Performance:** Numba JIT compilation + NumPy vectorization
- **Visual Quality:** 1920×1080 resolution with supersampling
- **Deep Zoom:** Zooms from 4.0 to 1e-5 width (10^5× magnification)
- **Psychedelic Colors:** Dynamic color cycling through rainbow palettes
- **Flexible Targets:** Configurable zoom centers (defaults to Seahorse Valley)
- **Progress Tracking:** Real-time rendering progress with ETA

## Default Configuration

| Parameter | Value | Description |
|---|---|---|
| **Resolution** | 1920×1080 | Full HD output |
| **Duration** | 60 seconds | At 30 FPS = 1800 frames |
| **Zoom Range** | 4.0 → 1e-5 | 100,000× magnification |
| **Target** | Seahorse Valley | (-1.7497591, 0.0000001) |
| **Iterations** | 1000 base | Adaptive based on zoom depth |
| **Palette** | psychedelic_rainbow | Color-cycling palette |

## Sample Output

The generator produces smooth zoom videos revealing the infinite detail of the Mandelbrot set:

1. **Wide view** → Classic Mandelbrot shape
2. **First zoom** → Bulbs and tendrils emerge  
3. **Deep zoom** → Self-similar mini-Mandelbrot copies
4. **Ultra-deep** → Seahorse valley spirals and infinite complexity

Each frame is color-cycled for a hypnotic psychedelic effect.

## Performance

- **Numba JIT:** Near C-speed computation of Mandelbrot iterations
- **Vectorized:** NumPy arrays for efficient pixel operations  
- **Memory-optimized:** Processes frames individually to avoid RAM limits
- **Progress bars:** Real-time ETA and frame rate monitoring

Typical render time on modern hardware: ~2-5 minutes for full 60-second video.

## Requirements

```
numpy>=1.24.0     # Vectorized computation
numba>=0.58.0     # JIT compilation for speed
pillow>=10.0.0    # Image processing
tqdm>=4.65.0      # Progress bars
torch>=2.0.0      # Tensor operations
```

## Architecture

```
mandelbrot_zoom/
├── core/         # Mandelbrot computation engine
├── rendering/    # Frame generation & zoom scheduling  
├── color/        # Palette management & color cycling
└── video/        # Frame assembly & video output
```

## Usage Examples

```bash
# Quick preview (10 frames)
python main.py --preview

# Custom location: zoom into the "Misiurewicz point"
python main.py --center-real -0.77568377 --center-imag 0.13646737

# Shorter, faster video
python main.py --duration 30 --fps 60

# Higher quality with more iterations
python main.py --base-max-iter 2000 --supersample-factor 4
```

## License

MIT