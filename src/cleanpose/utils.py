from pathlib import Path


def get_folders_with_patterns(folder, patterns=[]):
    folder = Path(folder)

    folders = set()

    for file in folder.rglob('*'):
        if file.is_file() and (file.parent in folders or not matches(str(file), patterns)):
            continue
        folders.add(file.parent)

    return folders


def matches(string, patterns):
    for pattern in patterns:
        if pattern not in string:
            return False
    return True
