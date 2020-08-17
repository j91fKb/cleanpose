import numpy as np


def interpolate(file, savefile):
    data = np.load(file)
    interp_data = np.zeros((data.shape[0], data.shape[1], 2))
    for i in range(data.shape[1]):
        last_x = data[0, i, 0]
        if np.isnan(last_x):
            last_x = 0
        last_y = data[0, i, 1]
        if np.isnan(last_y):
            last_y = 0
        interp_data[0, i, 0] = last_x
        interp_data[0, i, 1] = last_y
        for f in range(1, data.shape[0]):
            if np.isnan(data[f, i, 0]):
                interp_data[f, i, 0] = last_x
            else:
                interp_data[f, i, 0] = data[f, i, 0]
            if np.isnan(data[f, i, 1]):
                interp_data[f, i, 1] = last_y
            else:
                interp_data[f, i, 1] = data[f, i, 1]

            last_x = interp_data[f, i, 0]
            last_y = interp_data[f, i, 1]
    np.save(savefile, interp_data)
