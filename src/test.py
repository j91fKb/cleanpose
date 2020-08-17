from cleanpose import interpolate
from pathlib import Path
from tqdm import tqdm

folder = '/media/moneylab/VideoAnnotation/OpenPose/subjects/MG125/research/20190507_171001/cam1/01_filtered_pose'
folder = Path(folder)
save_folder = '/media/moneylab/VideoAnnotation/OpenPose/subjects/MG125/research/20190507_171001/cam1/01b_filtered_pose_interp'
save_folder = Path(save_folder)

files = [f for f in folder.glob('*.npy')]
for file in tqdm(files):
    interpolate(file, save_folder / file.name)