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


def parse_color(color_str):
    """Parse color string in format 'r,g,b' to tuple (r, g, b)"""
    try:
        r, g, b = map(int, color_str.split(','))
        return (r, g, b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid color format: {color_str}. Use 'r,g,b' format (e.g., '255,0,0')")


def create_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(description="Soccer Player and Ball Tracking System")
    
    # Video and model arguments
    parser.add_argument(
        "--video",
        default="videos/soccer_possession.mp4",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--ball_detection_model", 
        default=None, 
        type=str, 
        help="Path to the ball detection model"
    )
    parser.add_argument(
        "--player_detection_model", 
        default=None, 
        type=str, 
        help="Path to the player detection model"
    )
    
    # Feature toggles
    parser.add_argument(
        "--passes",
        action="store_true",
        help="Enable pass detection and visualization"
    )
    parser.add_argument(
        "--possession",
        action="store_true",
        help="Enable possession counter and tracking"
    )
    
    # Detection labels
    parser.add_argument(
        "--ball_label",
        default="ball",
        type=str,
        help="Ball label in YOLO model"
    )
    parser.add_argument(
        "--player_label",
        default="person",
        type=str,
        help="Player label in YOLO model"
    )
    
    # Team configuration
    parser.add_argument(
        "--first_team", 
        default="Chelsea", 
        type=str, 
        help="First team name"
    )
    parser.add_argument(
        "--second_team", 
        default="Man City", 
        type=str, 
        help="Second team name"
    )
    parser.add_argument(
        "--first_team_short", 
        default=None, 
        type=str, 
        help="First team short name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--second_team_short", 
        default=None, 
        type=str, 
        help="Second team short name (auto-generated if not provided)"
    )
    
    # Team colors (NEW FEATURE)
    parser.add_argument(
        "--first_team_color",
        default="255,0,0",  # Red
        type=parse_color,
        help="First team color in RGB format (e.g., '255,0,0' for red)"
    )
    parser.add_argument(
        "--second_team_color",
        default="0,0,255",  # Blue
        type=parse_color,
        help="Second team color in RGB format (e.g., '0,0,255' for blue)"
    )
    
    # Detection parameters
    parser.add_argument(
        "--player_image_size", 
        default=1280, 
        type=int, 
        help="Image size for player detection model"
    )
    parser.add_argument(
        "--ball_image_size", 
        default=1280, 
        type=int, 
        help="Image size for ball detection model"
    )
    parser.add_argument(
        "--player_confidence", 
        default=0.3, 
        type=float, 
        help="Confidence threshold for player detection (0.0-1.0)"
    )
    parser.add_argument(
        "--ball_confidence", 
        default=0.3, 
        type=float, 
        help="Confidence threshold for ball detection (0.0-1.0)"
    )
    
    # Tracking parameters (NEW FEATURE)
    parser.add_argument(
        "--player_distance_threshold",
        default=250,
        type=int,
        help="Distance threshold for player tracking"
    )
    parser.add_argument(
        "--ball_distance_threshold",
        default=150,
        type=int,
        help="Distance threshold for ball tracking"
    )
    parser.add_argument(
        "--classifier_inertia",
        default=20,
        type=int,
        help="Inertia value for the HSV classifier"
    )
    
    # Output options (NEW FEATURE)
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output video path (if not specified, overwrites input)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional information"
    )
    
    return parser


def validate_arguments(args):
    """Validate and normalize arguments"""
    # Validate confidence thresholds
    if not 0.0 <= args.player_confidence <= 1.0:
        raise ValueError("Player confidence must be between 0.0 and 1.0")
    if not 0.0 <= args.ball_confidence <= 1.0:
        raise ValueError("Ball confidence must be between 0.0 and 1.0")
    
    # Generate short names if not provided
    if args.first_team_short is None:
        args.first_team_short = args.first_team[:3].upper()
    if args.second_team_short is None:
        args.second_team_short = args.second_team[:3].upper()
    
    return args


def setup_detectors(args):
    """Initialize object detectors"""
    print("Initializing object detectors...")
    
    player_detector = (
        YoloV5(model_path=args.player_detection_model) 
        if args.player_detection_model 
        else YoloV5()
    )
    
    ball_detector = (
        YoloV5(model_path=args.ball_detection_model) 
        if args.ball_detection_model 
        else YoloV5()
    )
    
    return player_detector, ball_detector


def setup_classifier(inertia=20):
    """Initialize HSV classifier with inertia"""
    print(f"Initializing HSV classifier with inertia={inertia}...")
    hsv_classifier = HSVClassifier(filters=filters)
    return InertiaClassifier(classifier=hsv_classifier, inertia=inertia)


def setup_teams(args):
    """Initialize teams and match"""
    print(f"Setting up teams: {args.first_team} vs {args.second_team}")
    
    # Calculate board colors (slightly darker versions of main colors)
    def darken_color(color, factor=0.8):
        return tuple(int(c * factor) for c in color)
    
    team1 = Team(
        name=args.first_team,
        abbreviation=args.first_team_short,
        color=args.first_team_color,
        board_color=darken_color(args.first_team_color),
        text_color=(255, 255, 255),
    )
    
    team2 = Team(
        name=args.second_team,
        abbreviation=args.second_team_short,
        color=args.second_team_color,
        board_color=darken_color(args.second_team_color),
        text_color=(255, 255, 255),
    )
    
    return [team1, team2]


def setup_trackers(player_distance_threshold=250, ball_distance_threshold=150):
    """Initialize object trackers"""
    print("Setting up trackers...")
    
    player_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=player_distance_threshold,
        initialization_delay=3,
        hit_counter_max=90,
    )
    
    ball_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=ball_distance_threshold,
        initialization_delay=20,
        hit_counter_max=2000,
    )
    
    return player_tracker, ball_tracker


def main():
    # Parse and validate arguments
    parser = create_parser()
    args = parser.parse_args()
    args = validate_arguments(args)
    
    print("Starting Soccer Tracking System...")
    print(f"Video: {args.video}")
    print(f"Teams: {args.first_team} ({args.first_team_color}) vs {args.second_team} ({args.second_team_color})")
    
    # Initialize video
    video = Video(input_path=args.video, output_path=args.output)
    fps = video.video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Setup components
    player_detector, ball_detector = setup_detectors(args)
    classifier = setup_classifier(args.classifier_inertia)
    teams = setup_teams(args)
    player_tracker, ball_tracker = setup_trackers(
        args.player_distance_threshold, 
        args.ball_distance_threshold
    )
    
    # Initialize match
    match = Match(home=teams[0], away=teams[1], fps=fps)
    match.team_possession = teams[1]  # Start with second team
    
    # Initialize motion estimator and path
    motion_estimator = MotionEstimator()
    coord_transformations = None
    path = AbsolutePath()
    
    # Get background images for counters
    possession_background = match.get_possession_background()
    passes_background = match.get_passes_background()
    
    print("Starting frame processing...")
    frame_count = 0
    
    try:
        for frame in video:
            frame_count += 1
            
            if args.debug and frame_count % 30 == 0:  # Print every 30 frames
                print(f"Processing frame {frame_count}")
            
            # Get detections
            players_detections = get_player_detections(
                player_detector, frame, args.player_label, 
                args.player_image_size, args.player_confidence
            )
            ball_detections = get_ball_detections(
                ball_detector, frame, args.ball_label, 
                args.ball_image_size, args.ball_confidence
            )
            
            detections = ball_detections + players_detections
            
            # Update motion estimator
            coord_transformations = update_motion_estimator(
                motion_estimator=motion_estimator,
                detections=detections,
                frame=frame,
            )
            
            # Update trackers
            player_track_objects = player_tracker.update(
                detections=players_detections, 
                coord_transformations=coord_transformations
            )
            
            ball_track_objects = ball_tracker.update(
                detections=ball_detections, 
                coord_transformations=coord_transformations
            )
            
            # Convert tracked objects to detections
            player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
            ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
            
            # Classify players by team
            player_detections = classifier.predict_from_detections(
                detections=player_detections,
                img=frame,
            )
            
            # Update match state
            ball = get_main_ball(ball_detections)
            players = Player.from_detections(detections=player_detections, teams=teams)
            match.update(players, ball)
            
            # Ensure frame is in correct format before drawing
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # Ensure frame has correct shape (height, width, channels)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                if len(frame.shape) == 2:
                    # Grayscale to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:
                    # RGBA to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Convert to PIL for drawing
            frame_pil = PIL.Image.fromarray(frame)
            
            # Draw possession features
            if args.possession:
                frame_pil = Player.draw_players(
                    players=players, frame=frame_pil, confidence=False, id=True
                )
                
                if ball and ball.detection:
                    frame_pil = path.draw(
                        img=frame_pil,
                        detection=ball.detection,
                        coord_transformations=coord_transformations,
                        color=match.team_possession.color,
                    )
                
                frame_pil = match.draw_possession_counter(
                    frame_pil, counter_background=possession_background, debug=args.debug
                )
                
                if ball:
                    frame_pil = ball.draw(frame_pil)
            
            # Draw pass features
            if args.passes:
                pass_list = match.passes
                frame_pil = Pass.draw_pass_list(
                    img=frame_pil, passes=pass_list, coord_transformations=coord_transformations
                )
                
                frame_pil = match.draw_passes_counter(
                    frame_pil, counter_background=passes_background, debug=args.debug
                )
            
            # Convert back to numpy array with proper shape validation
            frame = np.array(frame_pil)
            
            # Final shape validation before writing
            if len(frame.shape) != 3:
                print(f"Warning: Frame has unexpected shape {frame.shape}, skipping frame {frame_count}")
                continue
                
            # Ensure proper data type
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Write frame to output video
            video.write(frame)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        print(f"Error occurred at frame {frame_count}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise
    finally:
        print(f"Processed {frame_count} frames")
        print("Processing complete!")


if __name__ == "__main__":
    main()