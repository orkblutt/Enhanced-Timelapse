# Enhanced Timelapse Generator

A sophisticated timelapse generator with AI-powered frame interpolation using TensorFlow Hub's FILM (Frame Interpolation for Large Motion) model. Create smooth, professional-quality timelapse videos with extensive customization options.

## Features

### 🎬 AI-Powered Frame Interpolation
- Uses Google's FILM model for smooth frame transitions
- Configurable interpolation levels (2^n intermediate frames)
- Dramatically reduces flickering and creates fluid motion

### 🔧 Extensive Customization
- **Frame Rate**: Custom FPS (1-120)
- **Resolution**: Any resolution (default 1920x1080)
- **Timing Control**: Configurable fade-in/out, hold durations
- **Effects**: Optional fade effects, stabilization
- **Output Formats**: Multiple video codecs (MP4V, H264, XVID)

### 🎯 Image Alignment & Stabilization
- **ECC Alignment**: Enhanced Correlation Coefficient method
- **ORB Alignment**: Feature-based alignment using ORB descriptors
- **Automatic Stabilization**: Reduce camera shake and movement

### ⚙️ Configuration Management
- **JSON Configuration**: Save and load settings
- **Command Line Interface**: Override any setting
- **Preview Mode**: Test settings with limited frames
- **Comprehensive Logging**: Track progress and debug issues

## Installation

### Requirements
- Python 3.8+
- TensorFlow >=2.8.0
- TensorFlow Hub >=0.12.0
- OpenCV (headless) >=4.5.0
- NumPy >=1.21.0
- tqdm >=4.64.0

### Install Dependencies

```bash
pip install tensorflow>=2.8.0 tensorflow-hub>=0.12.0 opencv-python-headless>=4.5.0 numpy>=1.21.0 tqdm>=4.64.0
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
python enhanced_timelapse.py images/
```

### Custom Settings

```bash
# Custom frame rate and resolution
python enhanced_timelapse.py images/ --fps 60 --width 3840 --height 2160

# Enable image alignment
python enhanced_timelapse.py images/ --align --alignment-method ecc

# Custom timing
python enhanced_timelapse.py images/ --fade-in 60 --fade-out 60 --hold 15

# Preview mode for testing
python enhanced_timelapse.py images/ --preview --preview-frames 50
```

### Configuration File

Create a configuration file for complex setups:

```bash
# Create sample config
python enhanced_timelapse.py --create-config

# Use config file
python enhanced_timelapse.py images/ --config my_config.json
```

## Configuration Options

### Video Settings
- `fps`: Output frame rate (default: 30)
- `width`: Video width (default: 1920)
- `height`: Video height (default: 1080)
- `codec`: Video codec (mp4v, h264, xvid)

### Timing Settings (in frames)
- `fade_in_frames`: Fade-in duration (default: 30)
- `fade_out_frames`: Fade-out duration (default: 30)
- `hold_frames`: Hold duration per image (default: 30)

### Interpolation Settings
- `interpolation_recursions`: Recursion level (default: 4)
- `enable_interpolation`: Enable/disable AI interpolation (default: true)

### Alignment Settings
- `enable_alignment`: Enable image alignment (default: false)
- `alignment_method`: Method (ecc, orb) (default: ecc)
- `alignment_reference_mode`: Reference selection (first, most_zoomed, middle) (default: first)
- `alignment_scale`: Scale factor for alignment (default: 0.25)

### Effects Settings
- `enable_fade_effects`: Enable fade in/out (default: true)
- `enable_stabilization`: Enable stabilization (default: false)

## Advanced Usage

### Sample Configuration File

```json
{
  "input_folder": "images",
  "output_file": "timelapse_4k.mp4",
  "fps": 60,
  "width": 3840,
  "height": 2160,
  "codec": "h264",
  "fade_in_frames": 90,
  "fade_out_frames": 90,
  "hold_frames": 20,
  "interpolation_recursions": 5,
  "enable_interpolation": true,
  "enable_alignment": true,
  "alignment_method": "ecc",
  "enable_fade_effects": true,
  "preview_mode": false,
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
}
```

### Command Line Examples

```bash
# High quality 4K timelapse
python enhanced_timelapse.py images/ --fps 60 --width 3840 --height 2160 --codec h264

# Fast preview with minimal effects
python enhanced_timelapse.py images/ --preview --no-fade --no-interpolation --hold 5

# Aligned and stabilized timelapse with smart reference selection
python enhanced_timelapse.py images/ --align --alignment-method ecc --alignment-reference most_zoomed --fps 30

# Custom interpolation level
python enhanced_timelapse.py images/ --interpolation-level 6 --fps 120

# Use most zoomed image as alignment reference for best detail preservation
python enhanced_timelapse.py images/ --align --alignment-reference most_zoomed --alignment-method ecc
```

## How It Works

### 1. Image Loading & Processing
- Automatically detects supported image formats
- Resizes images to target resolution
- Normalizes pixel values for processing

### 2. Optional Image Alignment
- **ECC Method**: Uses Enhanced Correlation Coefficient for affine transformations
- **ORB Method**: Feature-based alignment using ORB descriptors and RANSAC

### 3. AI Frame Interpolation
- Uses Google's FILM model from TensorFlow Hub
- Recursive interpolation creates smooth transitions
- Configurable recursion levels (2^n intermediate frames)

### 4. Effects Processing
- Fade-in/fade-out effects with black frames
- Configurable hold durations for each image
- Optional stabilization and cropping

### 5. Video Generation
- Multiple codec support (MP4V, H264, XVID)
- Configurable frame rates and resolutions
- Progress tracking with tqdm

## Performance Tips

### For Faster Processing
- Use `--preview` mode for testing
- Reduce `--interpolation-level` for fewer intermediate frames
- Disable `--align` if images are already aligned
- Use lower resolution for testing

### For Better Quality
- Enable `--align` for camera movement correction
- Use higher `--interpolation-level` for smoother motion
- Increase `--fps` for fluid playback
- Use `h264` codec for better compression

### Memory Optimization
- Process images in smaller batches
- Use GPU acceleration when available
- Monitor memory usage with large image sets

## Troubleshooting

### Common Issues

**"No GPU found" warning**
- The script works on CPU but GPU acceleration improves performance
- Install GPU-enabled TensorFlow for better performance

**"Alignment failed" messages**
- Normal for images with insufficient features
- Try different alignment methods or disable alignment

**Memory errors**
- Reduce image resolution or batch size
- Close other applications to free memory

**Codec not supported**
- Try different codecs (mp4v, h264, xvid)
- Install additional codecs if needed

### Debug Mode

```bash
python enhanced_timelapse.py images/ --verbose
```

Check the generated `timelapse.log` file for detailed information.

## File Structure

```
enhanced_timelapse.py    # Main script
README.md               # This file
requirements.txt        # Python dependencies
LICENSE                 # MIT License
examples/              # Example configurations
  ├── config_4k.json   # 4K configuration
  ├── config_fast.json # Fast processing
  └── config_hq.json   # High quality
scripts/               # Utility scripts
  └── batch_process.py # Batch processing script
```

## Examples

### 🎥 Real-World Example: Landscape Timelapse
Check out this landscape timelapse created with Enhanced Timelapse Generator:
**[Landscape Timelapse Example](https://youtu.be/LMSnMowcAXg)** - Demonstrates the AI-powered frame interpolation and alignment features in action.

### Nature Timelapse
```bash
python enhanced_timelapse.py nature_photos/ --fps 30 --align --fade-in 90 --fade-out 90
```

### Construction Progress
```bash
python enhanced_timelapse.py construction/ --fps 24 --interpolation-level 5 --hold 10
```

### Cloud Movement
```bash
python enhanced_timelapse.py clouds/ --fps 60 --no-fade --interpolation-level 6
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/yourusername/enhanced-timelapse.git
cd enhanced-timelapse
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Research for the FILM model (https://github.com/google-research/frame-interpolation?tab=readme-ov-file)
- TensorFlow Hub team for model hosting
- OpenCV community for image processing tools

## Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the generated log file (`timelapse.log`)
3. Open an issue on GitHub with details about your setup

## Changelog

### v2.0.0
- Added AI-powered frame interpolation
- Implemented multiple alignment methods
- Added configuration file support
- Enhanced logging and error handling
- Added preview mode for testing

### v1.0.0
- Initial release with basic timelapse functionality
