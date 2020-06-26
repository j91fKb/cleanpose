from cleanpose import RunOpenPose, OpenPoseVideo, ExtractPose, InitialPosition, FilterPose
from cleanpose.utils import get_folders_with_pattern

# RUN
openpose_dir = '/home/worker/openpose'
input_dir = '/home/worker/cleanpose/test/in'
output_dir = '/home/worker/cleanpose/test/out'

run = RunOpenPose(openpose_dir, input_dir, output_dir,
                  run_face=False, filters=[''])
run.initialize()
run.run()

# EXTRACT
folder = '/home/worker/cleanpose/test/out'
folders = get_folders_with_pattern(folder, pattern='.json')
for f in folders:
    extract = ExtractPose(f)
    extract.run()

# INITIAL POSITIONS
folder = '/home/worker/cleanpose/test/out'
folders = get_folders_with_pattern(folder, pattern='.json')
for f in folders:
    init_pos = InitialPosition(f)
    init_pos.run()

# FILTER
folder = '/home/worker/cleanpose/test/out'
folders = get_folders_with_pattern(folder, pattern='.json')
for f in folders:
    filter = FilterPose(f)
    filter.run()
