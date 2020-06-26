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


def calc_distance(p1, p2):
    return ((p1[:, 0] - p2[:, 0])**2 + (p1[:, 1] - p2[:, 1])**2)**.5


class FilterPose:
    def __init__(self, folder):
        matplotlib.use('Agg')
        plt.rcParams['figure.figsize'] = 10, 8

        self.folder = Path(folder)
        self.initial_positions_file = self.folder / 'initial_positions.csv'
        self.initial_positions = pd.read_csv(self.initial_positions_file)

        self.save_dir = self.folder / '01_filtered_pose'

    def run(self):
        if not self.save_dir.is_dir():
            os.mkdir(self.save_dir)

        n = len(self.initial_positions)
        print('Starting filtering')
        for i, video_name in enumerate(self.initial_positions['name']):
            print(f"file {i+1} of {n}")

            name = video_name.replace("_labeled.mp4", "")
            pose = self.get_pose(name)
            index_per_frame = self.find_subject(name, pose)
            selected_pose = self.get_selected_pose(index_per_frame, pose)
            fig = self.plot_results(name, selected_pose, pose)
            self.save_data(name, index_per_frame, selected_pose, pose)
            plt.close(fig)

    def df_index(self, name):
        return np.where(self.initial_positions['name'] == name)[0].item()

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

    def find_subject(self, name, pose):
        ind = self.df_index(name)
        origin = self.initial_positions.loc[ind]

        index_per_frame = np.zeros(pose.shape[0])
        index_per_frame[:] = np.nan

        frames_labeled = []
        weight = .95
        distance_limit = 50

        prev = origin.values[1:].astype('float64')
        count = 0
        labeled_ind = 10
        for i, p in enumerate(pose):
            if np.nansum(p[:, 0, 0]) == 0.0:
                continue

            if count < pose.shape[0]:  # labeled_ind
                head = p[:, 0, :2]
                distance = np.sqrt(
                    (head[:, 0] - prev[0])**2 + (head[:, 1] - prev[1])**2)
                smallest_distance = np.nanmin(distance)

                if smallest_distance < distance_limit:
                    subject_index = np.where(
                        distance == smallest_distance)[0][0]
                    index_per_frame[i] = subject_index

                    prev = p[subject_index, 0, :2].reshape(-1)
                    count += 1
                    frames_labeled.append(i)

                continue

            # Need to adjust this
            distance = np.zeros(p.shape[0])
            distance[:] = np.nan
            for ii, pp in enumerate(p):
                head = pp[0, :2]
                if np.isnan(np.sum(head)):
                    continue
                comparison_frames = np.flip(
                    np.array(frames_labeled[labeled_ind-10:labeled_ind]))
                prev = pose[comparison_frames,
                            index_per_frame[comparison_frames].astype(int), 0]
                frame_weights = np.power(
                    weight, i - comparison_frames) * prev[:, 2]

                distance[ii] = np.sum(
                    np.sqrt((head[0] - prev[:, 0])**2 + (head[1] - prev[:, 1])**2) * frame_weights)

            smallest_distance = np.nanmin(distance)

            if smallest_distance < distance_limit:
                subject_index = np.where(distance == smallest_distance)[0]
                if subject_index.size > 1:
                    subject_index = subject_index[0]
                index_per_frame[i] = subject_index

                frames_labeled.append(i)

        return index_per_frame

    def get_selected_pose(self, index_per_frame, pose):
        nan_inds = np.where(np.isnan(index_per_frame))[0]
        adjusted_index_per_frame = copy.copy(index_per_frame)
        adjusted_index_per_frame[nan_inds] = 0

        frames = [i for i in range(pose.shape[0])]

        return pose[frames, adjusted_index_per_frame.astype(int)]

    def plot_results(self, name, selected_pose, pose):
        fig = plt.figure()
        fig.suptitle(name)

        ax1 = plt.subplot('223')
        ax1.plot(selected_pose[:, 0, 0], label='x')
        ax1.plot(selected_pose[:, 0, 1], label='y')
        ax1.legend()
        ax1.set_xlabel("frame")
        ax1.set_ylabel("pixel")

        cap = cv2.VideoCapture(str(self.folder / (name + "_labeled.mp4")))

        ax2 = plt.subplot('224')
        s, frame = cap.read()
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(frame)
        ax2.plot(pose[:, :, 0, 0], pose[:, :, 0, 1], 'b+')
        ax2.plot(selected_pose[:, 0, 0], selected_pose[:, 0, 1], 'go')
        legend_elements = [Line2D([], [], marker='+', markerfacecolor='b', linestyle='None', label='all'),
                           Line2D([], [], marker='o', markerfacecolor='g', linestyle='None', label='subject')]
        ax2.legend(handles=legend_elements)

        frames = cap.get(7)
        check_frames = [int(i * frames / 8) for i in range(8)]

        axes = [plt.subplot(f"44{i}") for i in range(1, 9)]

        for i in range(8):
            ax = axes[i]
            frame_ind = check_frames[i]
            point = selected_pose[frame_ind, 0, :2]
            # while np.isnan(np.sum(point)):
            #     frame_ind += 1
            #     point = selected_pose[frame_ind, 0, :2]
            cap.set(1, frame_ind)
            s, frame = cap.read()
            self.plot_overlay_face(ax, point, frame, frame_ind + 1)

        return fig

    def plot_overlay_face(self, ax, point, frame, frame_num):
        x = point[0]
        y = point[1]

        if np.isnan(x) or np.isnan(y):
            return

        height = frame.shape[0]
        width = frame.shape[1]

        if x - 25 < 0:
            a1 = 0
            a2 = 50
        elif x + 25 > width - 1:
            a1 = width - 50
            a2 = width
        else:
            a1 = int(x - 25)
            a2 = a1 + 50

        if y - 25 < 0:
            b1 = 0
            b2 = 50
        elif y + 25 > height - 1:
            b1 = height - 50
            b2 = height
        else:
            b1 = int(y - 25)
            b2 = b1 + 50

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"frame: {frame_num}")
        ax.imshow(frame[b1:b2, a1:a2], extent=[a1, a2, b1, b2])
        ax.plot(point[0], point[1], 'ko', markersize=20, alpha=0.3)
        ax.plot(point[0], point[1], 'rx', markersize=10)

    def mk_save_dir(self):
        if not self.save_dir.is_dir():
            os.mkdir(self.save_dir)

    def save_data(self, name, index_per_frame, selected_pose=None, pose=None):
        self.mk_save_dir()
        if not (self.save_dir / 'filter_info').is_dir():
            (self.save_dir / 'filter_info').mkdir()

        file = self.save_dir / 'filter_info' / (name + "_filter_results.png")
        plt.savefig(file)

        file = self.save_dir / 'filter_info' / (name + "_subject_index")
        np.save(file, index_per_frame)

        if selected_pose is not None:
            file = self.save_dir / name
            np.save(file, selected_pose)
