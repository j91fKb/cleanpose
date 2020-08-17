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
from tqdm import tqdm


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

        files = natsorted([f for f in self.filter_dir.parent.glob('*.npy')])
        for f in tqdm(files):
            # print(f'file {i + 1} out of {len(files)}')
            name = f.stem

            data = np.load(self.data_dir / f'{name}.npy')
            indexes = np.load(self.filter_dir / f'{name}_subject_index.npy')

            if data.size == 0 or indexes.size == 0:
                continue

            nan_indexes = np.isnan(indexes)
            indexes[nan_indexes] = 0
            indexes = indexes.astype('int')
            frames = np.arange(data.shape[0], dtype='int')

            filtered_data = data[frames, indexes, :, :]

            save_file = self.save_dir / f'{name}.npy'
            np.save(save_file, filtered_data)
