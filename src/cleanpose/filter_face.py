import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
from natsort import natsorted
import copy


class FilterFace:
    def __init__(self, folder):
        matplotlib.use('Agg')
        plt.rcParams['figure.figsize'] = 10, 8

        self.folder = Path(folder)
        self.data_dir = self.folder / '00_raw_face'
        self.filter_dir = self.folder / '01_filtered_pose' / 'filter_info'
        self.save_dir = self.folder / '01_filtered_face'

    def run(self):
        if not self.save_dir.is_dir():
            os.mkdir(self.save_dir)

        files = natsorted([f for f in self.data_dir.glob('*.npy')])
        for i, f in enumerate(files):
            print(f'file {i + 1} out of {len(files)}')
            name = f.stem

            data = np.load(f)
            indexes = np.load(self.filter_dir / f'{name}_subject_index.npy')

            if data.size == 0 or indexes.size == 0:
                continue

            nan_indexes = np.isnan(indexes)
            indexes[nan_indexes] = 0
            indexes = indexes.astype('int')
            frames = np.arange(data.shape[0], dtype='int')

            filtered_data = data[frames, indexes, :, :]
            filtered_data[nan_indexes] = np.nan

            save_file = self.save_dir / f'{name}.npy'
            np.save(save_file, filtered_data)
