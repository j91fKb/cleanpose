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
from cleanpose import RunOpenPose, OpenPoseVideo, ExtractPose, InitialPosition, FilterPose
from cleanpose.utils import get_folders_with_pattern

# RUN
openpose_dir = '/openpose'
input_dir = '/test/in'
output_dir = '/test/out'

run = RunOpenPose(openpose_dir, input_dir, output_dir,
                  run_face=False, filters=[''])
run.initialize()
run.run()

# EXTRACT
folder = '/test/out'
folders = get_folders_with_pattern(folder, pattern='.json')
for f in folders:
    extract = ExtractPose(f)
    extract.run()

# INITIAL POSITIONS
folder = '/test/out'
folders = get_folders_with_pattern(folder, pattern='.json')
for f in folders:
    init_pos = InitialPosition(f)
    init_pos.run()

# FILTER
folder = '/test/out'
folders = get_folders_with_pattern(folder, pattern='.json')
for f in folders:
    filter = FilterPose(f)
    filter.run()
```