import numpy as np
import matplotlib.pyplot as plt

def extract_measurement(data, key):
    measurement = []
    for entry in data:
        if key == 'force':
            measurement.append(np.linalg.norm(entry[key]))
        elif key == 'position':
            measurement.append(entry[key][-1])
        else:
            measurement.append(entry[key])
    return measurement

def plot_force_and_z_pos(data):
    force = extract_measurement(data, 'force')
    z_pos = extract_measurement(data, 'position')

    plt.figure()
    plt.title('force')
    plt.grid()
    plt.plot(force)

    plt.figure()
    plt.title('z-pos')
    plt.plot(z_pos)
    plt.grid()
    plt.show()

def slice_data(data, sampling_location):
    # Could also be solved by slicing at element whose value has changed from the last ten elements or so (prone to noise)
    if sampling_location == 'upper_right':
        return data[:9368]
    if sampling_location == 'upper_left':
        return data[:7218]
    if sampling_location == 'center':
        return data[:6934]
    if sampling_location == 'lower_right':
        return data[:9201]
    if sampling_location == 'lower_left':
        return data[:6259]

data_upper_right = np.load('calibration_data/data_upper_right.npy', allow_pickle=True, encoding='latin1')
data_upper_left = np.load('calibration_data/data_upper_left.npy', allow_pickle=True, encoding='latin1')
data_center = np.load('calibration_data/data_centre.npy', allow_pickle=True, encoding='latin1')
data_lower_right = np.load('calibration_data/data_lower_right.npy', allow_pickle=True, encoding='latin1')
data_lower_left = np.load('calibration_data/data_lower_left.npy', allow_pickle=True, encoding='latin1')

data_upper_right = slice_data(data_upper_right, 'upper_right')
data_upper_left = slice_data(data_upper_left, 'upper_left')
data_center = slice_data(data_center, 'center')
data_lower_right = slice_data(data_lower_right, 'lower_right')
data_lower_left = slice_data(data_lower_left, 'lower_left')
