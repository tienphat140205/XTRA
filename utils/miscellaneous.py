from datetime import datetime
import os


def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()

    # Convert it to a string
    datetime_string = current_datetime.strftime(
        "%Y-%m-%d_%H-%M-%S")  # Format as YYYY-MM-DD HH:MM:SS
    return datetime_string


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)

