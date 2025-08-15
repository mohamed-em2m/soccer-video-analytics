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
white = { "name": "white", "lower_hsv": (0, 0, 184), "upper_hsv": (179, 39, 255), } red = { "name": "red", "lower_hsv": (0, 100, 0), "upper_hsv": (8, 255, 255), } blueish_red = { "name": "blueish_red", "lower_hsv": (170, 0, 0), "upper_hsv": (178, 255, 255), } orange = { "name": "orange", "lower_hsv": (7, 178, 0), "upper_hsv": (15, 255, 255), } yellow = { "name": "yellow", "lower_hsv": (23, 0, 0), "upper_hsv": (29, 255, 255), } green = { "name": "green", "lower_hsv": (48, 50, 0), "upper_hsv": (55, 255, 255), } sky_blue = { "name": "sky_blue", "lower_hsv": (95, 38, 0), "upper_hsv": (111, 190, 255), } blue = { "name": "blue", "lower_hsv": (112, 80, 0), "upper_hsv": (126, 255, 255), } black = { "name": "black", "lower_hsv": (0, 0, 0), "upper_hsv": (179, 255, 49), } all = [white, red, orange, yellow, green, sky_blue, blue, blueish_red, black]

