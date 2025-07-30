def format_time(seconds):
    """Format time in minutes and seconds."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} minutes {seconds} seconds"