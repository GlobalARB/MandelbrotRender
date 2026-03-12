# Mandelbrot Deep Zoom Video Generator

**High-performance tool for generating stunning 1080p videos zooming deep into the Mandelbrot set with psychedelic color cycling.**

Features Numba JIT compilation for near-C performance, supersampling anti-aliasing, and configurable zoom targets including the famous "Seahorse Valley" region.

![Mandelbrot Set](https://upload.wikimedia.org/wikipedia/commons/2/21/Mandel_zoom_00_mandelbrot_set.jpg)

## ✨ Features

- **High Performance:** Numba JIT compilation + NumPy vectorization for fast rendering
- **Visual Quality:** 1920×1080 resolution with 2x supersampling anti-aliasing
- **Deep Zoom:** Exponential zoom from 4.0 to 1e-5 width (100,000× magnification)
- **Psychedelic Colors:** Dynamic color cycling through customizable palettes
- **Flexible Targets:** Configurable zoom centers (defaults to Seahorse Valley mini-Mandelbrot)
- **Progress Tracking:** Real-time rendering progress with ETA
- **FFmpeg Integration:** High-quality H.264 encoding with web-optimized output

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nycarb/MandelbrotRender.git
cd MandelbrotRender

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- FFmpeg (for video encoding)

**Install FFmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Generate Your First Video

```bash
# Generate full 60-second video (1800 frames, ~2-5 min render time)
python main.py

# Quick 5-second preview (150 frames, ~10-30 sec render time)
python main.py --preview

# Custom zoom location (Misiurewicz point)
python main.py --center-real -0.77568377 --center-imag 0.13646737 --duration 30
```

Output video saved to: `./output/videos/mandelbrot_zoom.mp4`

## 📋 Default Configuration

| Parameter | Value | Description |
|---|---|---|
| **Resolution** | 1920×1080 | Full HD output |
| **Duration** | 60 seconds | At 30 FPS = 1800 frames |
| **Zoom Range** | 4.0 → 1e-5 | 100,000× magnification |
| **Target** | Seahorse Valley | (-1.7497591, 0.0000001) |
| **Iterations** | 1000 base | Adaptive scaling with zoom depth |
| **Supersampling** | 2× | Anti-aliasing for smooth edges |
| **Palette** | psychedelic_rainbow | RGB color cycling |

## 🎬 Sample Output

The generator produces smooth zoom videos revealing the infinite self-similar detail of the Mandelbrot set:

1. **Wide view** (0-10s) → Classic Mandelbrot bulb shape
2. **First zoom** (10-25s) → Tendrils and spiral arms emerge  
3. **Deep zoom** (25-45s) → Self-similar mini-Mandelbrot copies appear
4. **Ultra-deep** (45-60s) → Seahorse valley spirals and fractal complexity

Each frame cycles through psychedelic rainbow colors for hypnotic visual effect.

**10 snapshot images are automatically saved** during rendering to `./output/snapshots/` for trajectory verification.

## ⚡ Performance

- **Numba JIT:** Near C-speed computation of Mandelbrot iterations
- **Parallel Processing:** Multi-core CPU utilization via Numba's prange
- **Vectorized:** NumPy arrays for efficient pixel operations  
- **Memory-optimized:** Processes frames individually to avoid RAM limits
- **Adaptive Iterations:** Dynamically scales max_iter based on zoom depth

**Typical render times** (Apple M2, 8 cores):
- Preview (150 frames): ~15 seconds
- Full video (1800 frames): ~3 minutes

## 🎨 Usage Examples

### Preview Mode
```bash
# Fast 5-second preview
python main.py --preview
```

### Custom Zoom Locations

```bash
# Seahorse Valley (default - spectacular mini-Mandelbrot)
python main.py --center-real -1.7497591 --center-imag 0.0000001

# Misiurewicz point (intricate spirals)
python main.py --center-real -0.77568377 --center-imag 0.13646737

# Feigenbaum point (period-doubling region)
python main.py --center-real -1.401155 --center-imag 0.0
```

### Video Duration & Quality

```bash
# Shorter 30-second video
python main.py --duration 30

# 60 FPS high framerate
python main.py --fps 60 --duration 30

# Higher quality with more iterations
python main.py --max-iter 2000 --supersample 4
```

### Color Palettes

```bash
# Electric plasma palette
python main.py --palette electric_plasma

# Deep ocean blues
python main.py --palette deep_ocean

# Fire and ice
python main.py --palette fire_ice

# Available: psychedelic_rainbow, electric_plasma, deep_ocean, 
#            fire_ice, neon_dreams, sunset_glow
```

### Advanced Settings

```bash
# Extreme deep zoom (WARNING: 10^15 magnification, needs high max-iter)
python main.py --final-width 1e-15 --max-iter 5000 --duration 90

# Save individual frames as PNG
python main.py --save-frames

# Custom resolution (4K)
python main.py --width 3840 --height 2160 --supersample 1
```

## 📁 Project Structure

```
MandelbrotRender/
├── main.py                    # Entry point with CLI
├── requirements.txt           # Python dependencies
├── mandelbrot_zoom/
│   ├── core/                  # Mandelbrot computation engine
│   │   └── mandelbrot.py     # Numba JIT-compiled iteration kernel
│   ├── rendering/             # Frame rendering pipeline
│   │   └── frame_renderer.py # Anti-aliasing & zoom scheduling
│   ├── color/                 # Color palette system
│   │   └── palette.py        # Cosine gradient color mapping
│   └── video/                 # Video assembly
│       └── assembler.py      # FFmpeg integration
└── output/
    ├── videos/               # Output MP4 files
    ├── snapshots/            # Sample PNG frames
    └── frames/               # Individual frames (if --save-frames)
```

## 🔬 Technical Details

### Smooth Iteration Coloring

Uses the smooth iteration count algorithm for continuous gradients:

```
smooth_iter = n + 1 - log(log(|z|)) / log(2)
```

This eliminates banding artifacts and produces fluid color transitions.

### Exponential Zoom Schedule

Zoom width follows exponential decay for perceptually constant zoom speed:

```python
log_widths = np.linspace(log(initial), log(final), num_frames)
widths = np.exp(log_widths)
```

### Adaptive Iteration Scaling

Deeper zooms reveal finer detail requiring more iterations:

```python
zoom_factor = 4.0 / current_width
extra_iter = int(100 * log10(zoom_factor + 1))
max_iter = min(base_max_iter + extra_iter, 10000)
```

### Cosine Palette Generation

Colors generated procedurally using Inigo Quilez's cosine gradient technique:

```
color(t) = a + b * cos(2π(c*t + d))
```

Allows smooth color cycling and psychedelic effects.

## 🛠️ Command Line Options

```bash
python main.py --help
```

**Main options:**
- `--preview`: Quick 5-second preview
- `--center-real`, `--center-imag`: Zoom target coordinates
- `--duration`: Video length in seconds
- `--fps`: Frames per second (default: 30)
- `--width`, `--height`: Resolution (default: 1920×1080)
- `--initial-width`, `--final-width`: Zoom range
- `--max-iter`: Base iteration count (default: 1000)
- `--supersample`: Anti-aliasing factor (default: 2)
- `--palette`: Color scheme (default: psychedelic_rainbow)
- `--save-frames`: Save individual PNG frames

## 📦 Dependencies

```
numpy>=1.24.0     # Vectorized computation
numba>=0.58.0     # JIT compilation for 10-100× speedup
pillow>=10.0.0    # Image processing
tqdm>=4.65.0      # Progress bars
torch>=2.0.0      # Tensor operations (used by palette system)
```

## 🎓 Interesting Zoom Locations

| Location | Coordinates | Description |
|---|---|---|
| Seahorse Valley | (-1.7497591, 0.0000001) | Classic mini-Mandelbrot spiral |
| Misiurewicz Point | (-0.77568377, 0.13646737) | Intricate spiral patterns |
| Feigenbaum Point | (-1.401155, 0.0) | Period-doubling cascade |
| Elephant Valley | (0.2549, 0.0005) | Elephant-trunk structures |
| Double Spiral | (-0.7269, 0.1889) | Dual spiral formation |

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Smooth coloring algorithm: [Wikipedia - Mandelbrot Set](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set)
- Cosine palette technique: [Inigo Quilez](https://iquilezles.org/articles/palettes/)
- Numba project for JIT compilation capabilities

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

---

**Made by ARB** • [GitHub](https://github.com/nycarb) • [Website](https://nycarb.com)
