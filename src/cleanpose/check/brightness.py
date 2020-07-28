import numpy as np
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path


def extract_brightness(file):
    cap = cv2.VideoCapture(file)
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
    for i, file in enumerate(files):
        print(f'file {i+1} out of {len(files)}')
        brightness = extract_brightness(file)
        np.save(output_folder / file.name, brightness)
