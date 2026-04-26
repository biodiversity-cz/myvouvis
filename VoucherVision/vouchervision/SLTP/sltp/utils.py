import os

def validate_dir(dir):
    # Convert dir to a list if it's not already a list
    if not isinstance(dir, list):
        dir = [dir]

    # Iterate through the list and create directories if they don't exist
    for d in dir:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)