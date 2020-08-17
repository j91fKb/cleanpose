"""
RUN: k = ExtractFace("path to your folder with the json files")
"""

import os
import json
import numpy as np
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm


class ExtractFace:
    def __init__(self, folder):

        self.folder = Path(folder)
        self.files = natsorted([f for f in os.listdir(
            self.folder) if "coordinates.json" in f])

        self.save_dir = self.folder / '00_raw_face'
        if not self.save_dir.is_dir():
            os.mkdir(self.save_dir)

    def run(self):
        n = len(self.files)
        for video_name in tqdm(self.files):
            if (i+1) % 10 == 0:
                print(f"file {i+1} of {n}")

            name = video_name.replace("_coordinates.json", "")
            pose = self.get_pose(name)
            self.save_file(pose, name)
        print('done')

    def get_pose(self, name):
        data = json.load(open(self.folder / (name + "_coordinates.json")))

        n_ppl = np.zeros(len(data))
        for i, f in enumerate(data):
            n_ppl[i] = len(f['people'])
        max_ppl = np.max(n_ppl)

        pose = np.zeros((len(data), int(max_ppl), 70, 3))

        for i, f in enumerate(data):
            for k, p in enumerate(f['people']):
                pose[i, k] = np.reshape(p['face_keypoints_2d'], (70, 3))

        return pose

    def save_file(self, pose, name):
        savefile = self.save_dir / name
        np.save(savefile, pose)
