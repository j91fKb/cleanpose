import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path


class Smooth:
    def __init__(self, folder, face=False):
        self.folder = Path(folder)
        self.data_dir = self.folder / '01_filtered_pose'
        if face:
            self.data_dir = self.folder / '01_filtered_face'
        self.data_files = natsorted(list(self.data_dir.glob('*.npy')))
        self.save_dir = self.folder / '02_smoothed_pose'
        if face:
            self.save_dir = self.folder / '02_smoothed_face'

    def run(self):
        if not self.save_dir.is_dir():
            self.save_dir.mkdir()

        for i, file in enumerate(self.data_files):
            print(f'file {i + 1} of {len(self.data_files)}')

            data = np.load(file)
            data = np.transpose(data, (1, 2, 0))
            data = data[:, :2, :]

            smoothed_data = self.smooth(data)

            save_file = self.save_dir / file.name
            np.save(save_file, smoothed_data)

    def smooth(self, data):
        n_keypoints, n_coords, n_frames = data.shape

        x_pre = [[] for _ in range(n_frames)]
        s_pre = [[] for _ in range(n_frames)]
        x_pos = [[] for _ in range(n_frames)]
        s_pos = [[] for _ in range(n_frames)]
        s_n = 10 ** -3 * np.identity(n_keypoints)

        x_pre[0] = np.zeros((n_keypoints, n_coords))
        s_pre[0] = 10 ** -1 * np.identity(n_keypoints)
        x_pos[0] = x_pre[0] + \
            np.dot(np.dot(s_pre[0], np.linalg.pinv(
                s_pre[0] + s_n)), (data[:, :, 0] - x_pre[0]))
        s_pos[0] = s_pre[0] - \
            np.dot(np.dot(s_pre[0], np.linalg.pinv(
                s_pre[0] + s_n)), s_pre[0].T)

        for k in range(1, n_frames):
            x_pre[k] = x_pos[k - 1]
            s_pre[k] = s_pos[k - 1] + s_n

            x_pos[k] = x_pre[k] + \
                np.dot(np.dot(s_pre[k], np.linalg.pinv(
                    s_pre[k] + s_n)), (data[:, :, k] - x_pre[k]))
            s_pos[k] = s_pre[k] - \
                np.dot(np.dot(s_pre[k], np.linalg.pinv(
                    s_pre[k] + s_n)), s_pre[k].T)

        x_s = [[] for _ in range(n_frames)]
        x_s[-1] = x_pos[-1]

        for k in reversed(range(n_frames - 1)):
            G = np.dot(s_pos[k], np.linalg.pinv(s_pre[k + 1]))
            x_s[k] = x_pos[k] + np.dot(G, (x_s[k + 1] - x_pre[k + 1]))

        return np.array(x_s)
