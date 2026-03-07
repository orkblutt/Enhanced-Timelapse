# Alignment Tool Design

**Goal:** A standalone image alignment tool (`scripts/align_tool.py`) that does improved automatic alignment with an interactive overlay UI for manual corrections, outputting aligned images ready for the timelapse pipeline.

## Architecture

Two-phase workflow:
1. **Auto-align** — SIFT-based feature matching with RANSAC homography, confidence scoring per image
2. **Manual review** — OpenCV Qt5 highgui overlay window where the user can drag/rotate/scale images that failed or need adjustment

The tool outputs aligned images to a folder. The timelapse generator then uses that folder directly (no `--align` needed).

## Auto-Alignment Improvements

**Current problems:**
- ORB has too few features and poor scale/rotation invariance
- ECC only handles small displacements (affine, no perspective)
- No confidence scoring — bad alignments silently pass through
- No outlier rejection beyond basic RANSAC

**New approach:**
- Use SIFT (scale/rotation invariant, much more robust for handheld photos)
- Lowe's ratio test for match filtering (reject ambiguous matches)
- RANSAC homography with strict reprojection threshold
- Confidence score based on: number of inliers, reprojection error, homography condition number
- Flag images below confidence threshold for manual review

## Manual Overlay UI

**Built with OpenCV highgui (Qt5 backend, no extra deps).**

**Window layout:**
- Reference image displayed as background
- Current image overlaid with adjustable opacity
- Status bar showing image index, confidence score, controls

**Interactions:**
- **Left-click drag** — translate the image
- **Right-click drag** — rotate (horizontal) and scale (vertical)
- **Mouse wheel** — adjust overlay opacity
- **Keyboard:**
  - `a/d` — fine translate left/right
  - `w/s` — fine translate up/down
  - `q/e` — fine rotate
  - `+/-` — fine scale
  - `Enter` — accept alignment
  - `r` — reset to auto-alignment result
  - `Backspace` — reset to original (no alignment)
  - `n/p` — next/previous image
  - `Esc` — save and quit

**Display:**
- Blended overlay of reference + current image
- Optional side-by-side view toggle (`Tab` key)

## Data Flow

```
images/ (raw)
    │
    ▼
align_tool.py --input images/ --output aligned/ --reference 0
    │
    ├─ Phase 1: Auto-align all images (SIFT + RANSAC)
    │  └─ Save transforms to aligned/alignment_data.json
    │
    ├─ Phase 2: Show UI for low-confidence images
    │  └─ User adjusts, transforms updated in JSON
    │
    └─ Phase 3: Write aligned images to aligned/

aligned/ (ready for timelapse)
    │
    ▼
enhanced_timelapse.py aligned/ --output timelapse.mp4
```

## alignment_data.json Format

```json
{
  "reference_image": "20250403_143728.jpg",
  "images": [
    {
      "filename": "20250404_123200.jpg",
      "transform": [[1.0, 0.0, 5.2], [0.0, 1.0, -3.1], [0.0, 0.0, 1.0]],
      "confidence": 0.92,
      "manual_adjusted": false
    }
  ]
}
```

## CLI Interface

```
python scripts/align_tool.py --input images/ --output aligned/
python scripts/align_tool.py --input images/ --output aligned/ --reference 5
python scripts/align_tool.py --input images/ --output aligned/ --auto-only
python scripts/align_tool.py --input images/ --output aligned/ --review-all
python scripts/align_tool.py --input images/ --output aligned/ --threshold 0.5
```

- `--reference N` — use image N as reference (default: 0)
- `--auto-only` — skip manual review, just auto-align
- `--review-all` — show UI for all images, not just low-confidence
- `--threshold` — confidence threshold below which images are flagged (default: 0.7)
- `--width/--height` — output resolution (default: 1920x1080)

## Dependencies

No new dependencies — uses OpenCV (SIFT, highgui with Qt5), NumPy, and tqdm already in requirements.txt.
