# Studio-Quality Portrait Converter

Transform raw human portrait images into professional studio-quality photos using advanced computer vision and image processing techniques.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10+-orange.svg)

## 📋 Overview

This project converts raw portrait images captured in uncontrolled conditions (mobile camera, low light, motion, cluttered background) into studio-quality professional portraits. The system applies multiple enhancement techniques while maintaining natural appearance and fast inference performance.

## ✨ Features

### Mandatory Enhancements

- ✅ **Motion Blur Removal**: Automatic detection and removal of motion blur using Laplacian variance detection and unsharp masking
- ✅ **Background Blur (Bokeh Effect)**: Professional portrait mode with adjustable intensity (light/medium/strong)
- ✅ **Face Clarity Enhancement**: Improves facial details using MediaPipe face detection and CLAHE
- ✅ **Sharpness & Contrast**: Advanced sharpening with unsharp masking and contrast enhancement
- ✅ **Natural Skin Texture Preservation**: Bilateral filtering with frequency separation to maintain skin texture
- ✅ **Facial Identity Preservation**: All enhancements are carefully tuned to preserve original facial features

### Additional Features

- Batch processing support
- Before/after comparison generation
- Command-line interface
- Performance benchmarking
- Demo video creation

## 🛠️ Technology Stack

- **OpenCV** (4.8+): Core image processing operations
- **MediaPipe** (0.10+): Fast and accurate face detection and person segmentation
- **NumPy** (1.24+): Numerical computations
- **SciPy** (1.11+): Advanced signal processing
- **Pillow** (10.0+): Image I/O
- **Matplotlib** (3.7+): Visualization support

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**

```bash
cd /path/to/fog
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Command Line Interface

#### Process a Single Image

```bash
python main.py input.jpg -o output.jpg
```

#### Process All Images in a Directory

```bash
python main.py input_images/ -o output_images/ --batch
```

#### Create Before/After Comparison

```bash
python main.py portrait.jpg -o enhanced.jpg --compare comparison.jpg
```

#### Custom Enhancement Settings

```bash
# Strong bokeh effect
python main.py input.jpg -o output.jpg --bokeh-intensity strong

# Skip specific enhancements
python main.py input.jpg -o output.jpg --no-deblur --no-sharpness

# Verbose mode (show processing details)
python main.py input.jpg -o output.jpg -v
```

### Python API

```python
from portrait_enhancer import PortraitEnhancer
import cv2

# Load image
image = cv2.imread('input.jpg')

# Process with enhancer
with PortraitEnhancer(verbose=True) as enhancer:
    result = enhancer.process_image(
        image,
        remove_blur=True,
        add_bokeh=True,
        enhance_face=True,
        boost_sharpness=True,
        bokeh_intensity='medium'
    )

# Save result
cv2.imwrite('output.jpg', result)
```

### Generate Demo Materials

```bash
# Generate all demo materials (comparisons, grid, video)
python demo.py

# Custom directories
python demo.py --input my_images/ --output enhanced/ --demo demos/

# Performance benchmark
python demo.py --benchmark sample.jpg
```

## 🏗️ Project Structure

```
fog/
├── portrait_enhancer/          # Core enhancement modules
│   ├── __init__.py
│   ├── blur_removal.py         # Motion blur detection and removal
│   ├── segmentation.py         # Person/background segmentation
│   ├── face_enhancement.py     # Face clarity and skin smoothing
│   ├── bokeh_effect.py         # Background blur effect
│   ├── sharpness.py            # Sharpness and contrast enhancement
│   └── pipeline.py             # Main processing pipeline
├── main.py                      # CLI interface
├── demo.py                      # Demo generation and benchmarking
├── requirements.txt             # Python dependencies
├── input_images/                # Place input images here
├── output_images/               # Enhanced images output
└── demo_outputs/                # Demo materials (comparisons, video)
```

## 🔬 Technical Details

### Processing Pipeline

The enhancement pipeline follows these steps in order:

1. **Motion Blur Removal**
   - Laplacian variance detection (threshold: 100)
   - Adaptive sharpening based on blur severity
   - Unsharp masking with configurable strength

2. **Person Segmentation**
   - MediaPipe Selfie Segmentation (model 1)
   - Morphological refinement (closing + opening)
   - Edge feathering with Gaussian blur

3. **Bokeh Effect**
   - Variable Gaussian blur on background
   - Intensity-based blur strength (15-35px)
   - Subtle background brightness boost

4. **Face Enhancement**
   - MediaPipe Face Detection
   - Bilateral filtering for skin smoothing
   - Frequency separation for texture preservation
   - CLAHE for local contrast enhancement

5. **Final Quality Boost**
   - Unsharp masking (amount: 1.3)
   - CLAHE on LAB L-channel (clip limit: 1.8)
   - Saturation enhancement (1.15x)
   - Tone curve adjustments

### Performance

- **Processing Time**: < 5 seconds per image on CPU (typical portrait size: 1920x1080)
- **Memory Usage**: Efficient pipeline with minimal memory overhead
- **GPU Acceleration**: Not required (MediaPipe models are CPU-optimized)

## 📊 Results

The system produces professional-looking portraits with:
- Smooth, blurred backgrounds (bokeh effect)
- Sharp, clear facial features
- Natural skin texture (no over-smoothing)
- Enhanced color and contrast
- Preserved facial identity

Sample results are available in the `demo_outputs/` directory after running the demo script.

## 🎬 Demo Video

A demo video showcasing input/output comparisons is available:
- **Local**: `demo_outputs/demo_video.mp4` (after running demo.py)
- **Google Drive**: [TO BE ADDED - Upload demo_outputs/demo_video.mp4 to Google Drive and paste link here]

> **Note**: To complete the assignment, upload `demo_outputs/demo_video.mp4` to Google Drive, set sharing to "Anyone with the link", and update this README with the link.

## 🧪 Testing

### Quick Test

```bash
# Test all modules
python -c "from portrait_enhancer import blur_removal; blur_removal.test()"
python -c "from portrait_enhancer import segmentation; segmentation.test()"
python -c "from portrait_enhancer import face_enhancement; face_enhancement.test()"
python -c "from portrait_enhancer import bokeh_effect; bokeh_effect.test()"
python -c "from portrait_enhancer import sharpness; sharpness.test()"
python -c "from portrait_enhancer import pipeline; pipeline.test()"
```

### Performance Benchmark

```bash
python demo.py --benchmark input_images/sample.jpg
```

## 📝 CLI Options

```
usage: main.py [-h] -o OUTPUT [--batch] [--compare PATH] [--no-deblur]
               [--no-bokeh] [--no-face-enhance] [--no-sharpness]
               [--bokeh-intensity {light,medium,strong}] [-v]
               input

positional arguments:
  input                 Input image path or directory

optional arguments:
  -h, --help            Show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output image path or directory
  --batch               Process all images in input directory
  --compare PATH        Create before/after comparison image
  --no-deblur           Skip motion blur removal
  --no-bokeh            Skip background blur effect
  --no-face-enhance     Skip face enhancement
  --no-sharpness        Skip sharpness/contrast enhancement
  --bokeh-intensity {light,medium,strong}
                        Bokeh effect intensity (default: medium)
  -v, --verbose         Show detailed processing information
```

## 🤝 Contributing

This project was created as part of the FOG Technologies Machine Learning Engineer assignment.

## 📄 License

This project is created for educational and demonstration purposes.

## 👨‍💻 Author

**Deepak Kumar**

## 🙏 Acknowledgments

- **MediaPipe**: For providing efficient face detection and segmentation models
- **OpenCV**: For comprehensive image processing capabilities
- **FOG Technologies**: For the opportunity to work on this challenging problem

---

**Note**: For best results, use portrait images with clearly visible faces. The system works best with:
- Single person portraits or group photos
- Reasonably lit images (not extremely dark)
- Images where the person is the main subject

