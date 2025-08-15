color_ranges = [
    {
        "name": "white",
        "lower_hsv": (0, 0, 200),
        "upper_hsv": (179, 40, 255),
    },
    {
        "name": "black",
        "lower_hsv": (0, 0, 0),
        "upper_hsv": (179, 255, 50),
    },
    # Red (two ranges because it wraps around hue = 0)
    {
        "name": "red1",
        "lower_hsv": (0, 100, 50),
        "upper_hsv": (10, 255, 255),
    },
    {
        "name": "red2",
        "lower_hsv": (170, 100, 50),
        "upper_hsv": (179, 255, 255),
    },
    {
        "name": "orange",
        "lower_hsv": (11, 100, 50),
        "upper_hsv": (25, 255, 255),
    },
    {
        "name": "yellow",
        "lower_hsv": (26, 100, 50),
        "upper_hsv": (35, 255, 255),
    },
    {
        "name": "lime",
        "lower_hsv": (36, 100, 50),
        "upper_hsv": (45, 255, 255),
    },
    {
        "name": "green",
        "lower_hsv": (46, 100, 50),
        "upper_hsv": (85, 255, 255),
    },
    {
        "name": "cyan",
        "lower_hsv": (86, 100, 50),
        "upper_hsv": (95, 255, 255),
    },
    {
        "name": "sky_blue",
        "lower_hsv": (96, 80, 50),
        "upper_hsv": (110, 255, 255),
    },
    {
        "name": "blue",
        "lower_hsv": (111, 100, 50),
        "upper_hsv": (130, 255, 255),
    },
    {
        "name": "purple",
        "lower_hsv": (131, 100, 50),
        "upper_hsv": (145, 255, 255),
    },
    {
        "name": "magenta",
        "lower_hsv": (146, 100, 50),
        "upper_hsv": (169, 255, 255),
    },
    {
        "name": "pink",
        "lower_hsv": (160, 50, 150),
        "upper_hsv": (179, 150, 255),
    },
    {
        "name": "brown",
        "lower_hsv": (10, 100, 20),
        "upper_hsv": (20, 255, 200),
    },
]

# If you want them all in one dictionary by name:
color_dict = {c["name"]: c for c in color_ranges}

