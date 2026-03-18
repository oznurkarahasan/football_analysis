# Football Analysis — C++ Port

Real-time football video analysis using YOLOv8 object detection, ByteTrack multi-object tracking, and Lucas-Kanade optical flow for camera motion compensation.

# to run system

```bash
cd cpp_port
cmake -S . -B build && cmake --build build -j && ./build/football_analysis_app ./models/football_yolov8s.onnx ./test.mp4 ./output_test_cpp.mp4

```


## Requirements

- CMake >= 3.16
- C++17 compiler (GCC 9+ or Clang 10+)
- OpenCV 4.x with modules: `core`, `imgproc`, `videoio`, `dnn`, `video`

### Install OpenCV (Ubuntu/Debian)

```bash
sudo apt install libopencv-dev
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The binary is created at `build/football_analysis_app`.

## Usage

```
./build/football_analysis_app <model.onnx> <input_video> [output_video]
                              [ball_id player_id goalkeeper_id referee_id]
                              [x1 y1 x2 y2 x3 y3 x4 y4]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `model.onnx` | Yes | YOLOv8 ONNX model file |
| `input_video` | Yes | Input video file |
| `output_video` | No | Output video path (default: `output_test_cpp.mp4`) |
| `ball_id player_id goalkeeper_id referee_id` | No | Class IDs from the model (default: `0 2 1 3`) |
| `x1 y1 x2 y2 x3 y3 x4 y4` | No | 4 pixel corner points of the pitch for homography (world coordinates overlay) |

### Class IDs

Class IDs depend on how the model was trained. Check `data.yaml` from your training dataset:

```yaml
names:
  0: ball        # ballId = 0
  1: goalkeeper  # goalkeeperId = 1
  2: player      # playerId = 2
  3: referee     # refereeId = 3
```

## Examples

### Minimal — detection and tracking only

```bash
./build/football_analysis_app models/football_yolov8n.onnx test.mp4
```

### With custom output path

```bash
./build/football_analysis_app models/football_yolov8n.onnx test.mp4 output.mp4
```

### With custom class IDs

```bash
./build/football_analysis_app models/football_yolov8n.onnx test.mp4 output.mp4 0 2 1 3
```

### With homography (world coordinate overlay)

Provide 4 pixel corner points of the pitch visible in the video (bottom-left, top-left, top-right, bottom-right):

```bash
./build/football_analysis_app models/football_yolov8n.onnx test.mp4 output.mp4 \
    0 2 1 3 \
    110 1035  265 275  910 260  1640 915
```

> The 8 numbers are pixel coordinates of the 4 pitch corners in the video frame. These are **video-specific** — you must measure them for each different camera angle.

### 1280px model

If your model was exported with `imgsz=1280`, change `inputSize` in `src/main.cpp`:

```cpp
const int inputSize = 1280;  // line 421
```

Then rebuild:

```bash
cmake --build build -j
```

## Models

Place ONNX model files in the `models/` directory.

| Model | Export size | Speed (CPU) | Ball detection |
|---|---|---|---|
| `football_yolov8n.onnx` | 640px | Fast | Basic |
| `football_yolov8s_1280.onnx` | 1280px | Slow | Better |

### Export a model from Ultralytics

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
```

## Pipeline Overview

```
Pass 1  │  Read all frames + YOLO detection (no tracking yet)
        │
Pass 2  │  Camera motion estimation (Lucas-Kanade optical flow)
        │  ← uses left/right edge strips of each frame
        │
Pass 3  │  For each frame:
        │    detections − camera_motion → ByteTrack → + camera_motion
        │    → ball possession assignment
        │    → homography world coordinates (if enabled)
        │
Output  │  Annotated video with player IDs, ball marker, possession indicator
```

## Thresholds (src/main.cpp)

| Parameter | Default | Description |
|---|---|---|
| `confThreshold` | `0.10` | Minimum confidence for player/referee detection |
| `ballConfThreshold` | `0.30` | Minimum confidence for ball detection |
| `minBallSizePx` | `8` | Minimum ball bounding box size in pixels |
| `maxBallSizePx` | `80` | Maximum ball bounding box size in pixels |
| `inputSize` | `1280` | YOLO inference resolution — must match model export size |
