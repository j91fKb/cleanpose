"""
INPUT: Folder with labeled videos.
RUN: k = InitialPosition("path to video folder", "path to pose data folder")

Grab the initial positions of the patient in the video.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from natsort import natsorted
import sys
import time
import matplotlib
matplotlib.use('tkagg')


class InitialPosition:

    def __init__(self, folder):
        # plt.ion()
        plt.rcParams['figure.figsize'] = 10, 8

        self.folder = Path(folder)
        self.files = natsorted(
            [f for f in os.listdir(self.folder) if "labeled.mp4" in f])

        self.data_file = self.folder / "initial_positions.csv"
        self.data, previous = self.get_data()

        if not previous:
            self.pose_folder = self.folder / '00_raw_pose'
            self.grab_initial_positions()

        self.ind = 0

        self.fig = None
        self.ax = None
        self.img = None
        self.point = None

        # self.fig, self.ax = plt.subplots()

        # s, frame = self.get_frame()
        # self.img = self.ax.imshow(frame)
        # self.set_title()

        # self.point, = self.ax.plot([], [], 'go', markersize=20.0)
        # self.plot_point()

        # self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        # self.fig.canvas.mpl_connect('close_event', self.save_data)

    def run(self):
        self.fig, self.ax = plt.subplots()

        s, frame = self.get_frame()
        self.img = self.ax.imshow(frame)
        self.set_title()

        self.point, = self.ax.plot([], [], 'go', markersize=20.0)
        self.plot_point()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('close_event', self.save_data)

        plt.show()

    def on_click(self, event):
        if event.inaxes is not None:
            self.store_data(event)
            self.plot_point()
            plt.pause(0.1)
            self.update_ind()
            self.update_image()

    def on_keypress(self, event):
        sys.stdout.flush()
        if event.key == 'right':
            self.update_ind()
        elif event.key == 'left':
            self.update_ind(-1)
        self.update_image()

    def update_ind(self, change=1):
        self.ind = (self.ind + change) % len(self.files)

    def get_data(self):
        if self.data_file.is_file():
            print("Opening existing coordinates file.")
            return pd.read_csv(self.data_file), True
        else:
            df = pd.DataFrame(columns=["name", "x", "y"])
            df["name"] = [f.replace('_labeled.mp4', '') for f in self.files]
            return df, False

    def grab_initial_positions(self):
        print('Loading initial coordinates from pose.')
        for f in os.listdir(self.pose_folder):
            pose = np.load(self.pose_folder / f)

            name = f.replace('.npy', '')
            self.data.loc[self.data['name'] == name,
                          ['x', 'y']] = pose[0, 0, 0, :2]

    def plot_point(self):
        if self.data.iloc[self.ind, 1] and self.data.iloc[self.ind, 2]:
            self.point.set_data(
                self.data.iloc[self.ind, 1], self.data.iloc[self.ind, 2])
        else:
            self.point.set_data([], [])

    def store_data(self, event):
        self.data.loc[self.ind, ['x', 'y']] = [event.xdata, event.ydata]

    def save_data(self, event):
        self.data.to_csv(self.data_file, index=False)
        print('Saved coordinates.')
        # time.sleep(.1)
        # plt.close(fig=self.fig)

    def get_frame(self):
        vid = cv2.VideoCapture(str(self.folder / self.files[self.ind]))
        s, frame = vid.read()
        vid.release()
        return s, frame

    def update_image(self):
        s, frame = self.get_frame()
        self.img.set_data(frame)
        self.plot_point()
        self.set_title()
        plt.draw()

    def set_title(self):
        plt.title("file: " + str(self.ind + 1) + " / " + str(len(self.files)))
