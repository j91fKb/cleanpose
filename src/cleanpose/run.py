"""
Change file parameters at the top.
RUN: run()
"""

# pylint: disable=missing-function-docstring

import time
import os
import shutil
import subprocess
import traceback
import json
from pathlib import Path
from natsort import natsorted


# create directory
def create_dir(folder, parents=False):
    folder = Path(folder)
    if not folder.is_dir():
        folder.mkdir(parents=parents)


# clear folder contents
def clear_dir(folder):
    complete = True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            complete = False

    return complete


# convert video to mp4
def convert_to_mp4(file, output_file=None):
    if not output_file:
        output_file = Path(file).with_suffix('.mp4')

    cmd = ['ffmpeg', '-y', '-i', file, output_file]
    execute(cmd)

    return output_file


# print popen output
def print_popen_output(p):
    out, err = p.communicate()
    print(out.decode())
    if (err):
        print(err.decode())


# timed function
def timed(func):
    def wrapper():
        start = time.time()
        res = func()
        print(f"{time.time() - start} seconds")
        return res
    return wrapper


# execute a command
def execute(cmd):
    try:
        subprocess.run(cmd)
        # p = subprocess.Popen(
        # cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # for line in p.stdout:
        #     print(line)
        # while p.poll():
        #     print_popen_output(p)
        #     time.sleep(1)
        # p.wait()
        return 0
    except:
        # print_popen_output(p)
        # p.kill()
        # traceback.print_exc()
        return 1
    # else:
    #     print_popen_output(p)

    # return p.wait()


# get sub path to file
def get_sub_path(file, root):
    return str(Path(str(file).replace(str(root) + ('/' if str(root) != '/' else ''), '')).parent)


class OpenPoseVideo:
    def __init__(self, openpose_dir, video_file, output_dir, run_hand=False, run_face=False):
        self.openpose_dir = Path(openpose_dir)
        self.video_file = Path(video_file)
        self.output_dir = Path(output_dir)
        self.run_hand = run_hand
        self.run_face = run_face
        self.filter = filter

        self.openpose_bin = str(self.openpose_dir) + \
            '/build/examples/openpose/openpose.bin'
        self.output_file = Path(
            self.output_dir / f"{self.video_file.stem}_labeled.mp4")
        self.coords_dir = self.video_file.parent / 'coords'
        create_dir(self.coords_dir)
        self.coords_file = Path(
            self.output_dir / f"{self.video_file.stem}_coordinates.json")

    # run openpose
    def run(self):
        os.chdir(self.openpose_dir)
        cmd = [
            self.openpose_bin,
            '--video', self.video_file,
            '--write_video', self.output_file,
            '--write_json', self.coords_dir,
            '--display', '0',
            #    '--disable_blending',
        ]
        if self.run_hand:
            cmd += ['--hand']
        if self.run_face:
            cmd += ['--face']

        return execute(cmd) == 0

    # combine coordinates
    def combine_coordinates(self):
        coordinates = []

        for coord_file in natsorted(os.listdir(self.coords_dir)):
            contents = json.load(open(self.coords_dir / coord_file))
            coordinates.append(contents)

        json.dump(coordinates, open(self.coords_file, 'w'))


class RunOpenPose:
    def __init__(self, openpose_dir, input_dir, output_dir, video_types=['.avi'], run_face=False, run_hand=False, overwrite=False, filters=['']):
        self.openpose_dir = Path(openpose_dir)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.video_types = video_types
        self.run_face = run_face
        self.run_hand = run_hand
        self.overwrite = overwrite
        self.filters = filters

        self.openpose_bin = str(self.openpose_dir) + \
            '/build/examples/openpose/openpose.bin'
        self.cache = self.openpose_dir / 'tmp'

        self.files = []
        self.completed = []
        self.index = 0
        self.failed_files = []

    # search for videos
    def find_videos(self):
        for ext in self.video_types:
            self.files.extend([f for f in self.input_dir.rglob(
                '**/*' + ext) if any(pattern in str(f) for pattern in self.filters)])
        self.files = natsorted(self.files)
        self.completed = [False for _ in range(len(self.files))]
        return self.files

    # check if video is already labelled
    def is_labeled(self, file):
        file = Path(file)
        name = file.stem
        sub_path = get_sub_path(file, self.input_dir)

        labeled_file = self.output_dir / sub_path / f"{name}_labeled.mp4"
        coordinates_file = self.output_dir / \
            sub_path / f"{name}_coordinates.json"

        return labeled_file.is_file() and coordinates_file.is_file()

    # setup cache
    def setup_cache(self, file):
        file = Path(file)

        cache_dir = self.cache / file.stem
        create_dir(cache_dir)

        cache_file = cache_dir / file.name
        shutil.copyfile(file, cache_file)

        return cache_file, cache_dir

    # clear cache
    def clear_cache(self):
        clear_dir(self.cache)

    # convert to mp4
    def convert_to_mp4(self, file):
        file = Path(file)
        cmd = ['ffmpeg', '-y', '-i', file, file.with_suffix('.mp4')]
        return execute(cmd) == 0

    # initialize
    def initialize(self):
        create_dir(self.cache)
        self.find_videos()
        self.index = 0

    # run
    def run(self):
        while self.index < len(self.files):
            file = self.files[self.index]

            # check if complete or if not too overwrite
            if self.completed[self.index] or not self.overwrite and self.is_labeled(file):
                self.index += 1
                continue

            print(
                f"Video {self.index + 1} of {len(self.files)} -- {str(file)}")

            # setup cache and pathing
            cache_file, cache_dir = self.setup_cache(file)
            mp4_file = convert_to_mp4(cache_file)

            # run openpose
            openpose_video = OpenPoseVideo(
                self.openpose_dir, mp4_file, cache_dir, run_hand=self.run_hand, run_face=self.run_face)
            is_successful = openpose_video.run()

            if is_successful:
                # combine coordinates
                openpose_video.combine_coordinates()

                # copy files over
                output_dir = str(self.output_dir /
                                 get_sub_path(file, self.input_dir))
                create_dir(output_dir, parents=True)
                shutil.copy(str(mp4_file), output_dir)
                shutil.copy(str(openpose_video.output_file), output_dir)
                shutil.copy(str(openpose_video.coords_file), output_dir)
            else:
                self.failed_files.append((self.index, file))

            self.clear_cache()
            self.completed[self.index] = True
            self.index += 1
