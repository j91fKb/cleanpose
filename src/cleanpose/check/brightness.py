import numpy as np
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm


def extract_brightness(file):
    cap = cv2.VideoCapture(str(file))
    n_frames = cap.get(7)

    brightness = np.zeros((int(n_frames)))
    i = 0

    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness[i] = np.sum(frame)
        i += 1
        success, frame = cap.read()

    return brightness


def run_extract_brightness(input_folder, output_folder):
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)
    files = natsorted(
        [file for file in input_folder.rglob('**/*smoothed_pose*/*.npy')])
    for file in tqdm(files):
        brightness = extract_brightness(file.parent.parent / file.name.replace('.npy', '.mp4'))
        out_dir = output_folder / file.parent.parent.parent.name
        if not out_dir.is_dir():
            out_dir.mkdir()
        np.save(output_folder / file.parent.parent.parent.name / file.name, brightness)


if __name__ == "__main__":
    output_folder = '/media/moneylab/VideoAnnotation/OpenPose/check/brightness/MG125'
    input_folder = '/media/moneylab/VideoAnnotation/OpenPose/subjects/MG125/research'
    run_extract_brightness(input_folder, output_folder)