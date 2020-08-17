"""
RUN: k = ExtractPose("path to your folder with the json files")
"""

import os
import json
import numpy as np
from pathlib import Path
from natsort import natsorted
import h5py


class ExtractPoseHDF5:
    def __init__(self, folder):

        self.folder = Path(folder)
        self.files = natsorted([f for f in os.listdir(
            self.folder) if "coordinates.json" in f])

        self.save_dir = self.folder / '00_raw_pose'
        if not self.save_dir.is_dir():
            os.mkdir(self.save_dir)

    def run(self):
        n = len(self.files)
        for i, video_name in enumerate(self.files):
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

        pose = np.zeros((len(data), int(max_ppl), 25, 3))
        pose[:] = np.nan

        for i, f in enumerate(data):
            for k, p in enumerate(f['people']):
                pose[i, k] = np.reshape(p['pose_keypoints_2d'], (25, 3))

        return pose

    def save_file(self, pose, name):
        savefile = self.save_dir / name
        np.save(savefile, pose)


if __name__ == "__main__":
    f = h5py.File(
        '/media/moneylab/VideoAnnotation/OpenPose/check/file_format/hdf5/test.hdf5', 'w')
    file = '/media/moneylab/VideoAnnotation/OpenPose/MG120/all/research/20180928_171056/cam0/cam0_000000_coordinates.json'
    data = json.load(open(file))
    for i, frame in enumerate(data):
        frame_data = [person['pose_keypoints_2d']
                      for person in frame['people']]
        frame_data = np.reshape(frame_data, (len(frame['people']), 25, 3))
        f.create_dataset(str(i), data=frame_data)
    f.close()
