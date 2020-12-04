import numpy as np
import matplotlib.pyplot as plt

def extract_measurement(data, key):
    measurement = []
    for entry in data:
        if key == 'angular':
            measurement.append(np.linalg.norm(entry[key]))
        else:
            measurement.append(entry[key])
    return np.array(measurement)

def plot_force_and_z_pos(data, title):
    z_force = extract_measurement(data, 'force')[:,-1]
    z_pos = extract_measurement(data, 'position')[:,-1]

    plt.figure()
    plt.title('z_force' + ' - ' + title)
    plt.grid()
    plt.plot(z_force)

    plt.figure()
    plt.title('z_pos' + ' - ' + title)
    plt.plot(z_pos)
    plt.grid()
    plt.show()


def slice_data(data, sampling_location):
    # Values are manually read from the data
    if sampling_location == 'upper_right':
        return data[200:2111]
    if sampling_location == 'upper_left':
        return data[40:464]
    if sampling_location == 'center':
        return data[40:381]
    if sampling_location == 'lower_right':
        return data[40:2008]
    if sampling_location == 'lower_left':
        return data[40:174]


def remove_force_offset(data):
    z_offset = data[0]['force'][-1]
    for entry in data:
        entry['force'][-1] = entry['force'][-1] - z_offset


def calculate_y_values(data):
    y_values = []

    force = extract_measurement(data, 'force')
    position = extract_measurement(data, 'position')
    start_z_pos = position[0][-1]


    for i in range(1, len(force)): 
        z_force = force[i][-1]
        z_pos = position[i][-1]
        residual = start_z_pos - z_pos

        y_values.append(z_force / residual)

    return y_values

def calculate_x_values(data):
    x_values = []

    velocity = extract_measurement(data, 'linear')
    position = extract_measurement(data, 'position')
    start_z_pos = position[0][-1]

    for i in range(1, len(velocity)):
        z_vel = velocity[i][-1]
        z_pos = position[i][-1]
        residual = start_z_pos - z_pos

        x_values.append(z_vel / residual)

    return x_values

def calculate_calibration_curve(data):
    x = calculate_x_values(data)
    y = calculate_y_values(data)
    return x, y

def plot_calibration_curve(data, title):
    x, y = calculate_calibration_curve(data)

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('z_vel / r')
    plt.ylabel('z_force / r')
    plt.title('Calibration curve' + ' - ' + title)
    plt.grid()
    plt.show()


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

plot_calibration_curve(data_upper_left, 'upper_left')
plot_calibration_curve(data_center, 'center')
plot_calibration_curve(data_lower_right, 'lower_right')
plot_calibration_curve(data_lower_left, 'lower_left')