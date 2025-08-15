from inference.colors import  color_dict

def make_team_filter(team_name, color_names):
    """
    Create a team color filter for HSV detection.
    
    Args:
        team_name (str): Name of the team.
        color_names (list[str]): List of color names (must be in color_dict).
    
    Returns:
        dict: Team filter dictionary.
    """
    colors = []
    for name in color_names:
        if name not in color_dict:
            raise ValueError(f"Color '{name}' not found in color_dict.")
        colors.append(color_dict[name])
    
    return {
        "name": team_name,
        "colors": colors
    }

