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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--ball_detection_model", default=None, type=str, help="Path to the model"
)
parser.add_argument(
    "--player_detection_model", default=None, type=str, help="Path to the model"
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
args = parser.parse_args()


args = parser.parse_args()

first_team       = args.first_team
second_team      = args.second_team
first_team_short  = args.first_team_short  or first_team[:3].upper()
second_team_short = args.second_team_short or second_team[:3].upper()

player_label = args.player_label
ball_label = args.ball_label

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
player_detector = YoloV5(model_path= args.player_detection_model ) if args.player_detection_model else YoloV5()
ball_detector = YoloV5(model_path= args.ball_detection_model ) if args.ball_detection_model else YoloV5()

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
chelsea = Team(
    name=first_team,
    abbreviation=first_team_short,
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(   
                name=second_team,
                abbreviation=second_team_short,
                color=(240, 230, 188)
)
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city

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

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

for i, frame in enumerate(video):

    # Get Detections
    players_detections = get_player_detections(player_detector, frame, player_label)
    ball_detections = get_ball_detections(ball_detector, frame, ball_label)
    detections = ball_detections + players_detections

    # Update trackers
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

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)

    # Draw
    frame = PIL.Image.fromarray(frame)

    if args.possession:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
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
