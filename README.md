# Soccer Video Analytics


This repository contains the companion code of the blog post [Automatically measuring soccer ball possession with AI and video analytics](https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics) by [Tryolabs](https://tryolabs.com).

<a href="https://www.youtube.com/watch?v=CWnlGBVaRpQ" target="_blank">
<img src="https://user-images.githubusercontent.com/33181424/193869946-ad7e3973-a28e-4640-8494-bf899d5df3a7.png" width="60%" height="50%">
</a>

For further information on the implementation, please check out the post.

## How to install

To install the necessary dependencies we use [Poetry](https://python-poetry.org/docs). After you have it installed, follow these instructions:

1. Clone this repository:

   ```bash
   git clone git@github.com:tryolabs/soccer-video-analytics.git
   ```

2. Install the dependencies:

   ```bash
   poetry install
   ```

3. Optionally, download the ball.pt file [from the GitHub release](https://github.com/tryolabs/soccer-video-analytics/releases/tag/v0). Please note that this is just a toy model that overfits to a few videos, not a robust ball detection model.

## How to run

First, make sure to initialize your environment using `poetry shell`.

To run one of the applications (possession computation and passes counter) you need to use flags in the console.

These flags are defined in the following table:

| Argument               | Description                                      | Default value                  |
|------------------------|--------------------------------------------------|--------------------------------|
| --video                | Path to the input video                          | videos/soccer_possession.mp4  |
| --ball_detection_model | Path to the ball detection model (`.pt` format)  | None                           |
| --player_detection_model | Path to the player detection model (`.pt` format) | None                           |
| --passes               | Enable pass detection (flag)                     | False                          |
| --possession           | Enable possession counter (flag)                 | False                          |
| --ball_label           | Set ball label in YOLO model                     | ball                           |
| --player_label         | Set player label in YOLO model                   | person                         |
| --first_team           | First team name                                  | Chelsea                        |
| --second_team          | Second team name                                 | Man City                       |
| --first_team_short     | First team short name                            | None                           |
| --second_team_short    | Second team short name                           | None                           |


```
python run.py --<application> --model <path-to-the-model> --video <path-to-the-video>
```

>__Warning__: You have to run this command on the root of the project folder.

Here is an example on how to run the command:
    
```bash
python run.py --possession --ball_label models/ball.pt --video videos/soccer_possession.mp4
```

An mp4 video will be generated after the execution. The name is the same as the input video with the suffix `_out` added.
