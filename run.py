import argparse

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import make_team_filter  # we will build referee via this too
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--ball_detection_model", default=None, type=str, help="Path to the ball detection model"
)
parser.add_argument(
    "--player_detection_model", default=None, type=str, help="Path to the player detection model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
parser.add_argument(
    "--ball_label",
    default="ball",
    help="set ball label in yolo model",
)
parser.add_argument(
    "--player_label",
    default="person",
    help="set player label in yolo model",
)
parser.add_argument(
    "--player_image_size", default=640, type=int, help="Image size for player detection"
)
parser.add_argument(
    "--ball_image_size", default=640, type=int, help="Image size for ball detection"
)
parser.add_argument(
    "--player_confidence", default=0.5, type=float, help="Confidence threshold for player detection"
)
parser.add_argument(
    "--ball_confidence", default=0.5, type=float, help="Confidence threshold for ball detection"
)
parser.add_argument(
    "--first_team", default="Chelsea", type=str, help="First team Name"
)
parser.add_argument(
    "--second_team", default="Man City", type=str, help="Second team Name"
)
parser.add_argument(
    "--first_team_short", default=None, type=str, help="First team short name"
)
parser.add_argument(
    "--second_team_short", default=None, type=str, help="Second team short name"
)
parser.add_argument(
    "--first_team_color", default="255,0,0", type=str, help="First team color in RGB format (comma-separated)"
)
parser.add_argument(
    "--second_team_color", default="240,230,188", type=str, help="Second team color in RGB format (comma-separated)"
)
parser.add_argument(
    "--first_jesry_color", default="blue", type=str, help="first team jersey color name"
)
parser.add_argument(
    "--second_jesry_color", default="white", type=str, help="second team jersey color name"
)
parser.add_argument(
    "--output", default="output_video.mp4", type=str, help="Path to the output video file"
)

args = parser.parse_args()

first_team_name = args.first_team
second_team_name = args.second_team
first_team_short  = args.first_team_short  or first_team_name[:3].upper()
second_team_short = args.second_team_short or second_team_name[:3].upper()

# Parse UI colors (RGB)
first_team_color = tuple(map(int, args.first_team_color.split(',')))
second_team_color = tuple(map(int, args.second_team_color.split(',')))

print(f"First team ({first_team_name}) color: {first_team_color}")
print(f"Second team ({second_team_name}) color: {second_team_color}")

player_label = args.player_label
ball_label = args.ball_label

first_jesry_color  = args.first_jesry_color   # keep arg name as provided
second_jesry_color = args.second_jesry_color

video = Video(input_path=args.video, output_path=args.output)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
player_detector = YoloV5(model_path=args.player_detection_model) if args.player_detection_model else YoloV5()
ball_detector   = YoloV5(model_path=args.ball_detection_model)   if args.ball_detection_model   else YoloV5()

# Build HSV filters for jersey classification
# (pass a list of color names; if your make_team_filter accepts a single str it will still work)
first_team_filter  = make_team_filter(first_team_name,  first_jesry_color.split(","))
second_team_filter = make_team_filter(second_team_name, second_jesry_color.split(","))
referee_filter     = make_team_filter("Referee",        ["black"])

filters = [first_team_filter, second_team_filter, referee_filter]

# HSV Classifier (+ inertia)
hsv_classifier = HSVClassifier(filters=filters)
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match (for drawing/UI colors)
first_team = Team(
    name=first_team_name,
    abbreviation=first_team_short,
    color=first_team_color,
    board_color=(int(first_team_color[0]*0.8), int(first_team_color[1]*0.8), int(first_team_color[2]*0.8)),
    text_color=(255, 255, 255),
)
second_team = Team(
    name=second_team_name,
    abbreviation=second_team_short,
    color=second_team_color,
    board_color=(int(second_team_color[0]*0.8), int(second_team_color[1]*0.8), int(second_team_color[2]*0.8)),
    text_color=(255, 255, 255),
)
teams = [first_team, second_team]

match = Match(home=first_team, away=second_team, fps=fps)
match.team_possession = second_team  # initial possession (arbitrary)

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)
ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=20,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Drawer and backgrounds
path = AbsolutePath()
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

for i, frame in enumerate(video):
    # Detections
    players_detections = get_player_detections(player_detector, frame, player_label)
    ball_detections    = get_ball_detections(ball_detector, frame, ball_label)
    detections = ball_detections + players_detections

    # Update trackers & motion
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )
    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )
    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    # Convert tracked objects to detections
    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections   = Converter.TrackedObjects_to_Detections(ball_track_objects)

    # Classify player jerseys (adds .team to detections if possible)
    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # OPTIONAL: Map string team labels to Team objects if classifier returns strings
    for det in player_detections:
        if hasattr(det, "team") and isinstance(det.team, str):
            # match by full name or abbreviation
            if det.team == first_team.name or det.team == first_team.abbreviation:
                det.team = first_team
            elif det.team == second_team.name or det.team == second_team.abbreviation:
                det.team = second_team

    # Debug (every ~1s at 30fps)
    if i % 30 == 0:
        classified_count = sum(1 for det in player_detections if hasattr(det, 'team') and det.team is not None)
        print(f"Frame {i}: {classified_count}/{len(player_detections)} players classified")

    # Match update
    ball = get_main_ball(ball_detections)

    # IMPORTANT: use the CLASSIFIED detections to build Player objects
    players = Player.from_detections(detections=player_detections, teams=teams)

    try:
        match.update(players, ball)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'passes'" in str(e):
            print(f"Warning: Skipping pass detection for frame {i} due to team identification issue")
        else:
            raise

    # Draw
    frame = PIL.Image.fromarray(frame)

    if args.possession:
        frame = Player.draw_players(players=players, frame=frame, confidence=False, id=True)

        # Only draw ball path if ball exists
        frame = path.draw(
            img=frame,
            detection=ball.detection if ball else None,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

        if ball:
            frame = ball.draw(frame)

    if args.passes:
        pass_list = match.passes
        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )
        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    frame = np.array(frame)

    # Write video
    video.write(frame)
