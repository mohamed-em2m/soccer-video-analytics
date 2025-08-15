import argparse
import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass


# ======================
# Argument Parsing
# ======================
parser = argparse.ArgumentParser()

parser.add_argument("--video", default="videos/soccer_possession.mp4", type=str,
                    help="Path to the input video")
parser.add_argument("--output", default="output.mp4", type=str,
                    help="Path to save the output video")
parser.add_argument("--ball_detection_model", default=None, type=str,
                    help="Path to the YOLO model for ball detection")
parser.add_argument("--player_detection_model", default=None, type=str,
                    help="Path to the YOLO model for player detection")

parser.add_argument("--passes", action="store_true", help="Enable pass detection")
parser.add_argument("--possession", action="store_true", help="Enable possession counter")

parser.add_argument("--ball_label", default="ball", help="YOLO label name for ball")
parser.add_argument("--player_label", default="person", help="YOLO label name for player")

parser.add_argument("--first_team", default="Chelsea", type=str, help="First team name")
parser.add_argument("--second_team", default="Man City", type=str, help="Second team name")
parser.add_argument("--first_team_short", default=None, type=str, help="First team short name")
parser.add_argument("--second_team_short", default=None, type=str, help="Second team short name")

parser.add_argument("--first_team_color", default="255,0,0", type=str,
                    help="First team RGB color (comma-separated, e.g. 255,0,0)")
parser.add_argument("--second_team_color", default="255,255,255", type=str,
                    help="Second team RGB color (comma-separated, e.g. 255,255,255)")

parser.add_argument("--player_image_size", default=1280, type=int,
                    help="Image size for player detection")
parser.add_argument("--ball_image_size", default=1280, type=int,
                    help="Image size for ball detection")
parser.add_argument("--player_confidence", default=0.3, type=float,
                    help="Confidence threshold for players")
parser.add_argument("--ball_confidence", default=0.3, type=float,
                    help="Confidence threshold for balls")

args = parser.parse_args()


# ======================
# Helper: Parse RGB color strings
# ======================
def parse_rgb(color_str):
    return tuple(map(int, color_str.split(",")))


# ======================
# Assign CLI args
# ======================
first_team = args.first_team
second_team = args.second_team
first_team_short = args.first_team_short or first_team[:3].upper()
second_team_short = args.second_team_short or second_team[:3].upper()

first_team_color = parse_rgb(args.first_team_color)
second_team_color = parse_rgb(args.second_team_color)

player_image_size = args.player_image_size
ball_image_size = args.ball_image_size
player_label = args.player_label
ball_label = args.ball_label
ball_confidence = args.ball_confidence
player_confidence = args.player_confidence

# Video input & output
video = Video(input_path=args.video, output_path=args.output)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)


# ======================
# Object Detectors
# ======================
player_detector = YoloV5(model_path=args.player_detection_model) if args.player_detection_model else YoloV5()
ball_detector = YoloV5(model_path=args.ball_detection_model) if args.ball_detection_model else YoloV5()

# HSV Classifier with inertia
hsv_classifier = HSVClassifier(filters=filters)
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# ======================
# Teams & Match setup
# ======================
team_home = Team(
    name=first_team,
    abbreviation=first_team_short,
    color=first_team_color,
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
team_away = Team(
    name=second_team,
    abbreviation=second_team_short,
    color=second_team_color
)

teams = [team_home, team_away]
match = Match(home=team_home, away=team_away, fps=fps)
match.team_possession = team_away  # Initial possession assignment

# ======================
# Trackers & Motion Estimator
# ======================
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

# ======================
# Paths & Backgrounds
# ======================
path = AbsolutePath()
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()


# ======================
# Main Processing Loop
# ======================
for i, frame in enumerate(video):

    # Get detections
    players_detections = get_player_detections(player_detector, frame, player_label, player_image_size, player_confidence)
    ball_detections = get_ball_detections(ball_detector, frame, ball_label, ball_image_size, ball_confidence)
    detections = ball_detections + players_detections

    # Update trackers with motion estimation
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )
    player_track_objects = player_tracker.update(players_detections, coord_transformations)
    ball_track_objects = ball_tracker.update(ball_detections, coord_transformations)

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    # Classify players by team
    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Update match state
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)

    # Draw overlays
    frame = PIL.Image.fromarray(frame)

    if args.possession:
        frame = Player.draw_players(players=players, frame=frame, confidence=False, id=True)
        if ball:
            frame = path.draw(
                img=frame,
                detection=ball.detection,
                coord_transformations=coord_transformations,
                color=match.team_possession.color,
            )
        frame = match.draw_possession_counter(frame, counter_background=possession_background, debug=False)
        if ball:
            frame = ball.draw(frame)

    if args.passes:
        pass_list = match.passes
        frame = Pass.draw_pass_list(img=frame, passes=pass_list, coord_transformations=coord_transformations)
        frame = match.draw_passes_counter(frame, counter_background=passes_background, debug=False)

    frame = np.array(frame)

    # Write to output video
    video.write(frame)
