import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def make_plot(folder):
    folder = Path(folder)
    brightness = []
    for file in folder.glob('*.npy'):
        brightness.append(np.load(file))
    brightness = np.concatenate(brightness)
    x = np.arange(brightness.size) / (30 * 60 * 60)
    plt.figure(figsize=(12, 8))
    plt.xlabel('hours')
    plt.ylabel('brightness')
    plt.plot(x, brightness)
    plt.ylim(0, 80000000)
    return plt.gcf()


if __name__ == "__main__":
    check_dir = '/media/moneylab/VideoAnnotation/OpenPose/check/brightness'
    check_dir = Path(check_dir)
    output_dir = check_dir / 'figures'
    if not output_dir.is_dir():
        output_dir.mkdir()
    folders = set()
    for file in check_dir.rglob('**/*.npy'):
        folders.add(file.parent)
    for folder in tqdm(folders):
        fig = make_plot(folder)
        pt = folder.parent.name
        time = folder.name
        save_file = output_dir / f"{pt}--{time}.png"
        fig.savefig(save_file)
        plt.close(fig)