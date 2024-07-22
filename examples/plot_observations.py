import os
import pandas as pd
import yaml
import numpy as np

import matplotlib.pyplot as plt

def plot_trajectories(folder_path):
    """
    Plot the trajectories of a drone in 3D space.

    Args:
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        None
    """
    # Get all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Create a figure and axes for the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Load the gate configurations from the YAML file
    with open('config/getting_started.yaml', 'r') as file:
        gate_config = yaml.safe_load(file)

    # Extract the gate positions from the configuration
    gate_positions = gate_config['quadrotor_config']['gates']

    # Plot the gate positions in 3D
    for j, gate in enumerate(gate_positions):
        x, y, yaw = gate[0], gate[1], gate[5]
        half_length = 0.1875
        fr = []
        to = []
        height = 1 if gate[6] == 0 else 0.525

        delta_x = half_length * np.cos(yaw)
        delta_y = half_length * np.sin(yaw)

        for i in range(1, 4):
            fr.append([x + i * delta_x, y + i * delta_y, height - half_length])
            fr.append([x - i * delta_x, y - i * delta_y, height - half_length])
            to.append([x + i * delta_x, y + i * delta_y, height + half_length])
            to.append([x - i * delta_x, y - i * delta_y, height + half_length])

        fr = np.array(fr)
        to = np.array(to)
        ax.plot(fr[:, 0], fr[:, 1], fr[:, 2], color='blue')
        ax.plot(to[:, 0], to[:, 1], to[:, 2], color='blue')
        for i in range(4):
            ax.plot([fr[i][0], to[i][0]], [fr[i][1], to[i][1]], [fr[i][2], to[i][2]], color='blue')
        ax.text(x, y, height, f'Gate {j + 1}', color='black')

    for j, obstacle in enumerate(gate_config['quadrotor_config']['obstacles']):
        x, y = obstacle[0], obstacle[1]
        z = 1.05
        ax.plot([x, x], [y, y], [0, z], color='red')
        ax.text(x, y, z, f'Obstacle {j + 1}', color='black')

    # Iterate over each CSV file
    for file in csv_files:
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        x = df["drone_x"].to_numpy()[:-1]
        y = df["drone_y"].to_numpy()[:-1]
        z = df["drone_z"].to_numpy()[:-1]

        # Plot the x, y, and z trajectories in 3D
        if len(df) > 10:
            ax.plot(x, y, z)

    # Set labels and title for the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectories')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    folder_path = '/Users/michaelloncsek/Privat/TUM_Master/lsy_drone/lsy_drone_racing/trained_models/2024-06-03_10-00-58/episodes'
    plot_trajectories(folder_path)
