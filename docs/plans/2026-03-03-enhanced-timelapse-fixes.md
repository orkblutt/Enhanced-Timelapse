# Enhanced Timelapse Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all bugs, memory issues, unused code, performance problems, and design flaws in `enhanced_timelapse.py` identified during code review.

**Architecture:** Refactor `enhanced_timelapse.py` to stream frames directly to video (eliminating OOM), fix the recursive interpolation duplicate-frame bug, make CLI/config interaction correct, move TF initialization out of module scope, and clean up dead code. Also fix `scripts/batch_process.py` for compatibility.

**Tech Stack:** Python 3, TensorFlow, TensorFlow Hub (FILM model), OpenCV, NumPy, tqdm

**Note:** TensorFlow and OpenCV are not installed in the current environment. Testing will rely on syntax checks (`python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`) and manual code review. No pytest available.

---

### Task 1: Fix recursive generator duplicate frames

**Files:**
- Modify: `enhanced_timelapse.py:485-499`

**Step 1: Fix the recursive_generator method**

The current code yields `frame2` unconditionally at line 499, causing duplicates at every recursion level. Remove the trailing `yield frame2` and keep the base case yielding only `frame1`. The final frame of each transition is handled by the caller.

```python
def recursive_generator(self, frame1: np.ndarray, frame2: np.ndarray,
                      num_recursions: int, interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
    """Recursive generator for midpoint interpolation."""
    if num_recursions == 0:
        yield frame1
    else:
        dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(
            np.expand_dims(frame1, axis=0),
            np.expand_dims(frame2, axis=0),
            dt
        )[0]
        yield from self.recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator)
        yield from self.recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator)
```

**Step 2: Update generate_timelapse to handle the final frame correctly**

In the interpolation loop (around line 562-579), the generator now yields `[f1, ..., intermediates]` without the final `frame2`. Adjust the caller to use all frames from the generator (they are the intermediates between each pair) and add `img2` hold frames after:

```python
# In the interpolation branch:
for idx in tqdm(range(len(self.images) - 1), desc="Processing transitions"):
    img1 = self.images[idx]
    img2 = self.images[idx + 1]

    gen = self.recursive_generator(
        img1, img2, self.config.interpolation_recursions, self.interpolator
    )
    # Generator yields: [img1, interp1, interp2, ..., interpN] (no img2)
    # Skip the first frame (img1) since it was already added as hold or is the first image
    first = True
    for frame in gen:
        if first:
            first = True  # skip img1, already held
            first = False
            continue
        all_frames.append(frame)

    # Hold next image
    if self.config.hold_frames > 0:
        for _ in range(self.config.hold_frames):
            all_frames.append(img2)
```

Wait — this gets reworked in Task 5 (streaming). For now just fix the generator and leave the caller doing `list(gen)` with `transition_frames[1:]` (skip first only, no longer need to strip last since it's not yielded).

Corrected caller logic:
```python
gen = self.recursive_generator(
    img1, img2, self.config.interpolation_recursions, self.interpolator
)
transition_frames = list(gen)
# Skip first frame (img1, already held/added), keep all intermediates
intermediates = transition_frames[1:]
all_frames.extend(intermediates)
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`
Expected: No output (success)

**Step 4: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "fix: remove duplicate frames in recursive interpolation generator"
```

---

### Task 2: Fix CLI defaults overriding config file

**Files:**
- Modify: `enhanced_timelapse.py:664-728`

**Step 1: Change argparse defaults to None**

For all arguments that correspond to config fields, set `default=None` so we can detect whether the user explicitly passed them:

```python
# Video settings
parser.add_argument('--fps', type=int, default=None, help="Output frame rate (default: 30)")
parser.add_argument('--width', type=int, default=None, help="Output width (default: 1920)")
parser.add_argument('--height', type=int, default=None, help="Output height (default: 1080)")
parser.add_argument('--codec', default=None, help="Video codec (default: mp4v)")

# Timing settings
parser.add_argument('--fade-in', type=int, default=None, help="Fade-in duration in frames (default: 30)")
parser.add_argument('--fade-out', type=int, default=None, help="Fade-out duration in frames (default: 30)")
parser.add_argument('--hold', type=int, default=None, help="Hold duration per image in frames (default: 30)")

# Effects
parser.add_argument('--interpolation-level', type=int, default=None, help="Interpolation recursion level (default: 4)")

# Preview
parser.add_argument('--preview-frames', type=int, default=None, help="Number of preview frames (default: 100)")
```

**Step 2: Only override config when args are explicitly set**

Replace the block at lines 711-728 with conditional overrides:

```python
# Override config with explicitly provided command line arguments
config.input_folder = args.input_folder
if args.output:
    config.output_file = args.output
if args.fps is not None:
    config.fps = args.fps
if args.width is not None:
    config.width = args.width
if args.height is not None:
    config.height = args.height
if args.codec is not None:
    config.codec = args.codec
if args.fade_in is not None:
    config.fade_in_frames = args.fade_in
if args.fade_out is not None:
    config.fade_out_frames = args.fade_out
if args.hold is not None:
    config.hold_frames = args.hold
if args.no_fade:
    config.enable_fade_effects = False
if args.no_interpolation:
    config.enable_interpolation = False
if args.interpolation_level is not None:
    config.interpolation_recursions = args.interpolation_level
if args.align:
    config.enable_alignment = True
if args.alignment_method:
    config.alignment_method = args.alignment_method
if args.alignment_reference:
    config.alignment_reference_mode = args.alignment_reference
if args.preview:
    config.preview_mode = True
if args.preview_frames is not None:
    config.preview_frames = args.preview_frames
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 4: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "fix: CLI args no longer override config file defaults"
```

---

### Task 3: Fix division by zero on fade frames

**Files:**
- Modify: `enhanced_timelapse.py:543-548` (fade in) and `enhanced_timelapse.py:589-595` (fade out)

**Step 1: Guard against fade_frames <= 1**

```python
# Fade in
if self.config.enable_fade_effects and self.config.fade_in_frames > 0:
    logger.info("Generating fade-in effect...")
    for i in range(self.config.fade_in_frames):
        alpha = i / max(self.config.fade_in_frames - 1, 1)
        frame = black * (1 - alpha) + self.images[0] * alpha
        all_frames.append(frame)

# Fade out (same pattern)
if self.config.enable_fade_effects and self.config.fade_out_frames > 0:
    logger.info("Generating fade-out effect...")
    last_img = self.images[-1]
    for i in range(self.config.fade_out_frames):
        alpha = 1 - (i / max(self.config.fade_out_frames - 1, 1))
        frame = last_img * alpha + black * (1 - alpha)
        all_frames.append(frame)
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 3: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "fix: prevent division by zero when fade frames is 1"
```

---

### Task 4: Fix bare except and add VideoWriter cleanup

**Files:**
- Modify: `enhanced_timelapse.py:358` (bare except)
- Modify: `enhanced_timelapse.py:611-636` (VideoWriter cleanup)

**Step 1: Replace bare except with specific exception**

At line 358, change:
```python
except:
    continue
```
to:
```python
except Exception:
    continue
```

**Step 2: Add try/finally for VideoWriter**

```python
def _write_video(self, frames):
    """Write frames to video file."""
    if self.config.codec == 'h264':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    elif self.config.codec == 'xvid':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)

    video_writer = cv2.VideoWriter(
        self.config.output_file, fourcc, self.config.fps,
        (self.config.width, self.config.height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {self.config.output_file}")

    try:
        for frame in tqdm(frames, desc="Writing video"):
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
    finally:
        video_writer.release()
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 4: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "fix: replace bare except, add VideoWriter cleanup with try/finally"
```

---

### Task 5: Stream frames to video instead of accumulating in memory

**Files:**
- Modify: `enhanced_timelapse.py:501-636` (rewrite `generate_timelapse` and `_write_video`)

This is the biggest change. Restructure `generate_timelapse` to yield frames via a generator, and `_write_video` to consume them.

**Step 1: Create a frame generator method**

Replace the monolithic `generate_timelapse` with a `_generate_frames` generator that yields frames one-by-one, and a thin `generate_timelapse` that sets up, counts, and writes:

```python
def _generate_frames(self) -> Generator[np.ndarray, None, None]:
    """Generate all timelapse frames as a stream."""
    black = np.zeros((self.config.height, self.config.width, 3), dtype=np.float32)

    # Fade in
    if self.config.enable_fade_effects and self.config.fade_in_frames > 0:
        logger.info("Generating fade-in effect...")
        for i in range(self.config.fade_in_frames):
            alpha = i / max(self.config.fade_in_frames - 1, 1)
            yield black * (1 - alpha) + self.images[0] * alpha

    # Hold first image
    if self.config.hold_frames > 0:
        for _ in range(self.config.hold_frames):
            yield self.images[0]

    # Process transitions
    if self.config.enable_interpolation:
        logger.info("Processing image transitions with interpolation...")
        for idx in tqdm(range(len(self.images) - 1), desc="Processing transitions"):
            img1 = self.images[idx]
            img2 = self.images[idx + 1]

            gen = self.recursive_generator(
                img1, img2, self.config.interpolation_recursions, self.interpolator
            )
            first = True
            for frame in gen:
                if first:
                    first = False
                    continue  # skip img1
                yield frame

            # Hold next image
            for _ in range(self.config.hold_frames):
                yield img2
    else:
        logger.info("Processing without interpolation...")
        for idx in range(1, len(self.images)):
            for _ in range(self.config.hold_frames):
                yield self.images[idx]

    # Fade out
    if self.config.enable_fade_effects and self.config.fade_out_frames > 0:
        logger.info("Generating fade-out effect...")
        last_img = self.images[-1]
        for i in range(self.config.fade_out_frames):
            alpha = 1 - (i / max(self.config.fade_out_frames - 1, 1))
            yield last_img * alpha + black * (1 - alpha)

def generate_timelapse(self):
    """Generate the timelapse video."""
    logger.info("Starting timelapse generation...")

    self.load_images()

    # Load and resize images (source images still need to be in memory for interpolation)
    logger.info("Loading and resizing images...")
    raw_images = []
    for file_path in tqdm(self.image_files, desc="Loading images"):
        raw_images.append(self.load_and_resize_image(file_path))

    # Align images if enabled
    if self.config.enable_alignment:
        logger.info(f"Aligning images using {self.config.alignment_method} method...")
        logger.info(f"Reference selection mode: {self.config.alignment_reference_mode}")
        reference_idx = self.select_reference_image(raw_images)
        reference = raw_images[reference_idx]
        logger.info(f"Using image {reference_idx} as alignment reference")

        self.images = []
        for i, img in enumerate(tqdm(raw_images, desc="Aligning images")):
            if i == reference_idx:
                self.images.append(img)
            else:
                self.images.append(self.align_image_to_reference(img, reference))
    else:
        self.images = raw_images
        logger.info("Skipping image alignment")

    # Initialize interpolator if needed
    if self.config.enable_interpolation:
        logger.info("Initializing interpolator...")
        self.interpolator = Interpolator()

    # Stream frames to video
    frame_gen = self._generate_frames()

    # Apply preview limit
    if self.config.preview_mode:
        frame_gen = _limit_generator(frame_gen, self.config.preview_frames)
        logger.info(f"Preview mode: limiting to {self.config.preview_frames} frames")

    logger.info("Writing frames to video...")
    frame_count = self._write_video(frame_gen)

    logger.info(f"Timelapse generation completed! Output: {self.config.output_file}")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Duration: {frame_count / self.config.fps:.2f} seconds")
```

**Step 2: Add the helper and update _write_video to return count**

Add a module-level helper:
```python
def _limit_generator(gen, max_items):
    """Limit a generator to max_items yields."""
    for i, item in enumerate(gen):
        if i >= max_items:
            break
        yield item
```

Update `_write_video` to accept a generator and return count:
```python
def _write_video(self, frames) -> int:
    """Write frames to video file. Returns frame count."""
    if self.config.codec == 'h264':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    elif self.config.codec == 'xvid':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)

    video_writer = cv2.VideoWriter(
        self.config.output_file, fourcc, self.config.fps,
        (self.config.width, self.config.height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {self.config.output_file}")

    frame_count = 0
    try:
        for frame in tqdm(frames, desc="Writing video"):
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            frame_count += 1
    finally:
        video_writer.release()

    return frame_count
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 4: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "perf: stream frames to video writer instead of accumulating in memory"
```

---

### Task 6: Move TF initialization out of module scope

**Files:**
- Modify: `enhanced_timelapse.py:57-69` (module-level TF setup)
- Modify: `enhanced_timelapse.py:39-48` (module-level logging setup)

**Step 1: Remove module-level TF setup and wrap in function**

Remove lines 57-69 from module scope. Create a `_setup_tensorflow()` function and call it from `TimelapseGenerator.__init__`:

```python
# Remove from module scope:
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# ... etc

_tf_initialized = False

def _setup_tensorflow():
    """Configure TensorFlow environment. Called once on first use."""
    global _tf_initialized
    if _tf_initialized:
        return
    _tf_initialized = True

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
```

**Step 2: Call from TimelapseGenerator.__init__**

```python
def __init__(self, config: TimelapseConfig):
    _setup_tensorflow()
    self.config = config
    self.interpolator = None
    self.image_files = []
    self.images = []
```

**Step 3: Make logging setup lazy**

Replace the module-level `logging.basicConfig(...)` with a setup function. Keep the `logger = logging.getLogger(__name__)` at module level but defer handler setup:

```python
logger = logging.getLogger(__name__)

_logging_initialized = False

def _setup_logging(log_file: str = 'timelapse.log', verbose: bool = False):
    """Configure logging handlers. Called once."""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
```

Call `_setup_logging()` from `main()` before any work is done. Call `_setup_logging(verbose=True)` if `args.verbose`.

**Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 5: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "refactor: move TF init and logging setup out of module scope"
```

---

### Task 7: Remove unused config fields

**Files:**
- Modify: `enhanced_timelapse.py:72-116` (TimelapseConfig dataclass)
- Modify: `examples/config_fast.json`
- Modify: `examples/config_4k.json`
- Modify: `examples/config_hq.json`

**Step 1: Remove unused fields from TimelapseConfig**

Remove these fields:
- `quality: int = 95` (line 84)
- `batch_size: int = 1` (line 107)
- `enable_stabilization: bool = False` (line 103)
- `crop_to_stable: bool = False` (line 104)

**Step 2: Remove those keys from example config JSON files**

Remove `"quality"`, `"batch_size"`, `"enable_stabilization"`, `"crop_to_stable"` from each JSON file.

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 4: Commit**

```bash
git add enhanced_timelapse.py examples/
git commit -m "cleanup: remove unused config fields (quality, batch_size, stabilization)"
```

---

### Task 8: Precompute ORB features in homography analysis

**Files:**
- Modify: `enhanced_timelapse.py:304-375` (`find_reference_image_by_homography`)

**Step 1: Precompute all features once**

```python
def find_reference_image_by_homography(self, images: List[np.ndarray]) -> int:
    """Find most zoomed image using homography scale analysis."""
    if len(images) <= 1:
        return 0

    logger.info("Using homography analysis to find optimal reference...")

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Precompute features for all images
    features = []
    for img in tqdm(images, desc="Extracting features"):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        features.append((kp, des))

    # Calculate relative scales between image pairs
    scale_scores = []

    for i, (kp_i, des_i) in enumerate(features):
        if des_i is None:
            scale_scores.append(0.0)
            continue

        relative_scales = []

        for j, (kp_j, des_j) in enumerate(features):
            if i == j or des_j is None:
                continue

            matches = bf.match(des_i, des_j)

            if len(matches) < 10:
                continue

            src_pts = np.float32([kp_i[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_j[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                    scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
                    avg_scale = (scale_x + scale_y) / 2
                    relative_scales.append(avg_scale)
            except Exception:
                continue

        if relative_scales:
            avg_relative_scale = np.mean(relative_scales)
            scale_scores.append(1.0 / (avg_relative_scale + 0.1))
        else:
            scale_scores.append(0.0)

    most_zoomed_idx = np.argmax(scale_scores)
    logger.info(f"Homography analysis: most zoomed image at index {most_zoomed_idx}")

    return most_zoomed_idx
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`

**Step 3: Commit**

```bash
git add enhanced_timelapse.py
git commit -m "perf: precompute ORB features in homography reference analysis"
```

---

### Task 9: Update batch_process.py for compatibility

**Files:**
- Modify: `scripts/batch_process.py:27` (import)
- Modify: `scripts/batch_process.py:54` (config copy)

**Step 1: Verify batch_process.py still works with refactored imports**

The import `from enhanced_timelapse import TimelapseConfig, TimelapseGenerator` should still work since we moved TF init into `_setup_tensorflow()` which is only called when `TimelapseGenerator.__init__` runs. No changes needed to imports.

**Step 2: Fix config copy using dataclass**

Line 54 uses `TimelapseConfig(**config.__dict__)` which includes the `supported_formats` list. Since `__post_init__` sets it from None, passing the list directly is fine. However, use `dataclasses.asdict` + constructor for a proper deep copy:

```python
from dataclasses import asdict
# ...
folder_config = TimelapseConfig(**asdict(config))
```

Already imports `asdict` via `enhanced_timelapse`. Actually `batch_process.py` doesn't import asdict. Add:

```python
from enhanced_timelapse import TimelapseConfig, TimelapseGenerator
from dataclasses import asdict
```

And update line 54:
```python
folder_config = TimelapseConfig(**asdict(config))
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('scripts/batch_process.py').read())"`

**Step 4: Commit**

```bash
git add scripts/batch_process.py
git commit -m "fix: update batch_process.py for compatibility with refactored module"
```

---

### Task 10: Final verification and cleanup

**Step 1: Syntax check both files**

Run: `python3 -c "import ast; ast.parse(open('enhanced_timelapse.py').read())"`
Run: `python3 -c "import ast; ast.parse(open('scripts/batch_process.py').read())"`

**Step 2: Review the complete file for consistency**

Read through the full file to verify:
- No remaining references to removed config fields
- All method signatures consistent
- No orphaned code

**Step 3: Commit any final tweaks**

```bash
git add -A
git commit -m "chore: final cleanup and consistency pass"
```
