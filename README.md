# Soccer Video Analytics

This repository contains the companion code for automated soccer video analysis, featuring ball possession tracking, pass detection, and team classification using AI and computer vision techniques.

<a href="https://www.youtube.com/watch?v=CWnlGBVaRpQ" target="_blank">
<img src="https://user-images.githubusercontent.com/33181424/193869946-ad7e3973-a28e-4640-8494-bf899d5df3a7.png" width="60%" height="50%">
</a>

The system uses YOLO object detection models to identify players and balls, HSV-based jersey classification for team assignment, and advanced tracking algorithms to analyze game dynamics in real-time.

## Features

- **Ball possession tracking** with visual possession counter
- **Pass detection and counting** with trajectory visualization
- **Team classification** based on jersey colors using HSV filtering
- **Multi-object tracking** with motion compensation
- **Customizable team colors and names** for display
- **Flexible model configuration** for different detection scenarios

## Installation

To install the necessary dependencies we use [Poetry](https://python-poetry.org/docs). After you have it installed, follow these instructions:

1. Clone this repository:
   ```bash
   git clone git@github.com/tryolabs/soccer-video-analytics.git
   ```

2. Install the dependencies:
   ```bash
   poetry install
   ```

3. Optionally, download pre-trained models:
   - Ball detection model: `ball_tracking_model.pt`
   - Player detection model: `player_tracking_model.pt`

## Usage

First, make sure to initialize your environment using `poetry shell`.

### Command Line Arguments

| Argument                    | Description                                        | Default Value                 |
|----------------------------|----------------------------------------------------|------------------------------ |
| `--video`                  | Path to the input video                            | `videos/soccer_possession.mp4` |
| `--ball_detection_model`   | Path to the ball detection model (.pt format)     | `None` (uses default YOLOv5)  |
| `--player_detection_model` | Path to the player detection model (.pt format)   | `None` (uses default YOLOv5)  |
| `--passes`                 | Enable pass detection (flag)                       | `False`                       |
| `--possession`             | Enable possession counter (flag)                   | `False`                       |
| `--ball_label`             | Set ball label in YOLO model                       | `ball`                        |
| `--player_label`           | Set player label in YOLO model                     | `person`                      |
| `--player_image_size`      | Image size for player detection                    | `640`                         |
| `--ball_image_size`        | Image size for ball detection                      | `640`                         |
| `--player_confidence`      | Confidence threshold for player detection          | `0.5`                         |
| `--ball_confidence`        | Confidence threshold for ball detection            | `0.5`                         |
| `--first_team`             | First team name                                    | `Chelsea`                     |
| `--second_team`            | Second team name                                   | `Man City`                    |
| `--first_team_short`       | First team short name (auto-generated if not set) | `None`                        |
| `--second_team_short`      | Second team short name (auto-generated if not set)| `None`                        |
| `--first_team_color`       | First team color name in RGB format (comma-separated)  | `255,0,0`                     |
| `--second_team_color`      | Second team color name in RGB format (comma-separated) | `240,230,188`                 |
| `--first_jesry_color`      | First team jersey color name for classification   | `blue`                        |
| `--second_jesry_color`     | Second team jersey color name for classification  | `white`                       |
| `--output`                 | Path to the output video file                      | `output_video.mp4`            |

### Basic Usage

```bash
python run.py --possession --video <path-to-video>
```

### Advanced Example

```bash
python run.py \
    --possession \
    --passes \
    --video /path/to/input/video.mp4 \
    --output /path/to/output/processed.mp4 \
    --ball_detection_model models/ball_tracking_model.pt \
    --player_detection_model models/player_tracking_model.pt \
    --player_label player \
    --ball_label ball \
    --first_team "Barcelona" \
    --second_team "Real Madrid" \
    --first_team_short FCB \
    --second_team_short RMD \
    --first_jesry_color "blue,red" \
    --second_jesry_color "white" \
    --first_team_color "0,0,255" \
    --second_team_color "255,255,255" \
    --player_image_size 1280 \
    --ball_image_size 3840 \
    --player_confidence 0.2 \
    --ball_confidence 0.0
```

### Jersey Color Classification

The system supports multiple jersey colors per team. Use comma-separated color names for teams with multiple jersey colors:

- Single color: `--first_jesry_color "blue"`
- Multiple colors: `--first_jesry_color "blue,red"`

Supported color names include: `blue`, `red`, `white`, `black`, `yellow`, `green`, etc.

### Model Configuration

- **Default models**: If no model paths are specified, the system uses default YOLOv5 models
- **Custom models**: Provide paths to your trained `.pt` model files
- **Image sizes**: Larger image sizes (1280, 3840) provide better detection accuracy but slower processing
- **Confidence thresholds**: Lower values (0.0-0.2) capture more detections but may include false positives

## Output

The processed video will be saved to the specified output path with:
- Player tracking with team colors and IDs
- Ball trajectory visualization
- Possession counter (if enabled)
- Pass detection and counting (if enabled)

> **Warning**: Make sure to run commands from the root of the project folder.

## Technical Details

The system employs several advanced techniques:
- **YOLOv5** for object detection
- **HSV color space filtering** for jersey classification
- **Norfair tracking** with motion compensation
- **Inertia-based classification** to maintain consistent team assignments
- **Multi-object tracking** with configurable distance thresholds
