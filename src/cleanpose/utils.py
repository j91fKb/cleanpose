from pathlib import Path


def get_folders_with_pattern(folder, pattern=''):
    folder = Path(folder)

    folders = set()

    for file in folder.rglob(f"**/*{pattern}*"):
        folders.add(file.parent)

    return folders
