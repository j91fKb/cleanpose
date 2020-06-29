# cleanpose

Extract out a specific person from the openpose output.

## Requirements & Install

* [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* ffmpeg
* python: ^3.6

```
    cd <path to this folder>
    pip install -r requirements.txt
    pip install .
```

## Use

```python
from cleanpose import RunOpenPose, OpenPoseVideo, ExtractPose, InitialPosition, FilterPose, Smooth
from cleanpose.utils import get_folders_with_patterns

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
folders = get_folders_with_patterns(
    folder, patterns=['.json'])  # can specify multiple patterns
for f in folders:
    extract = ExtractPose(f)
    extract.run()

# INITIAL POSITIONS
for f in folders:
    init_pos = InitialPosition(f)
    init_pos.run()

# FILTER
for f in folders:
    filter = FilterPose(f)
    filter.run()

# SMOOTH
for f in folders:
    smooth = Smooth(f)
    smooth.run()
```