import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import yaml
import pandas as pd


def save_observations(obs_list, save_path, current_datetime):
    """
    Save the list of observations to a CSV file.

    Args:
        obs_list (list): List of observations.
        save_path (str): Path to save the CSV file.
        current_datetime (str): Current date and time.

    Returns:
        None
    """
    csv_file = save_path / f"observations_{current_datetime}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(obs_list[0].keys())
        for obs in obs_list:
            writer.writerow([
                obs['drone_x'],
                obs['drone_y'],
                obs['drone_z'],
                obs['drone_roll'],
                obs['drone_pitch'],
                obs['drone_yaw'],
                obs['drone_vx'],
                obs['drone_vy'],
                obs['drone_vz'],
                obs['drone_vroll'],
                obs['drone_vpitch'],
                obs['drone_vyaw'],
                obs['gate1_x'],
                obs['gate1_y'],
                obs['gate1_z'],
                obs['gate1_yaw'],
                obs['gate2_x'],
                obs['gate2_y'],
                obs['gate2_z'],
                obs['gate2_yaw'],
                obs['gate3_x'],
                obs['gate3_y'],
                obs['gate3_z'],
                obs['gate3_yaw'],
                obs['gate4_x'],
                obs['gate4_y'],
                obs['gate4_z'],
                obs['gate4_yaw'],
                obs['gate1_in_range'],
                obs['gate2_in_range'],
                obs['gate3_in_range'],
                obs['gate4_in_range'],
                obs['obstacle1_x'],
                obs['obstacle1_y'],
                obs['obstacle1_z'],
                obs['obstacle2_x'],
                obs['obstacle2_y'],
                obs['obstacle2_z'],
                obs['obstacle3_x'],
                obs['obstacle3_y'],
                obs['obstacle3_z'],
                obs['obstacle4_x'],
                obs['obstacle4_y'],
                obs['obstacle4_z'],
                obs['obstacle1_in_range'],
                obs['obstacle2_in_range'],
                obs['obstacle3_in_range'],
                obs['obstacle4_in_range'],
                obs['gate_id']
                ])
            
def process_observation(x, print_flag=False):
    """
    Process the observation data and extract relevant information.

    Args:
        x (ndarray): The input observation data.
        print_flag (bool, optional): Flag to indicate whether to print the extracted information. 
                                     Defaults to False.

    Returns:
        dict: A dictionary containing the extracted information from the observation data.
            - 'drone_pos' (ndarray): The position of the drone [x, y, z].
            - 'drone_rpy' (ndarray): The roll, pitch, and yaw angles of the drone.
            - 'drone_vxyz' (ndarray): The velocity of the drone in the x, y, and z directions.
            - 'drone_vrpy' (ndarray): The angular velocity of the drone in roll, pitch, and yaw.
            - 'gates_xyz_yaw' (dict): The positions and yaw angles of the gates.
                - 'gate1' (ndarray): The position and yaw angle of gate 1.
                - 'gate2' (ndarray): The position and yaw angle of gate 2.
                - 'gate3' (ndarray): The position and yaw angle of gate 3.
                - 'gate4' (ndarray): The position and yaw angle of gate 4.
            - 'gates_in_range' (ndarray): The gates that are in range.
            - 'obstacles_xyz' (dict): The positions of the obstacles.
                - 'obstacle1' (ndarray): The position of obstacle 1.
                - 'obstacle2' (ndarray): The position of obstacle 2.
                - 'obstacle3' (ndarray): The position of obstacle 3.
                - 'obstacle4' (ndarray): The position of obstacle 4.
            - 'obstacles_in_range' (ndarray): The obstacles that are in range.
            - 'gate_id' (float): The ID of the gate.

    """
    observations = {
        'drone_x': x[0][0],
        'drone_y': x[0][1],
        'drone_z': x[0][2],
        'drone_roll': x[0][3],
        'drone_pitch': x[0][4],
        'drone_yaw': x[0][5],
        'drone_vx': x[0][6],
        'drone_vy': x[0][7],
        'drone_vz': x[0][8],
        'drone_vroll': x[0][9],
        'drone_vpitch': x[0][10],
        'drone_vyaw': x[0][11],
        'gate1_x': x[0][12],
        'gate1_y': x[0][13],
        'gate1_z': x[0][14],
        'gate1_yaw': x[0][15],
        'gate2_x': x[0][16],
        'gate2_y': x[0][17],
        'gate2_z': x[0][18],
        'gate2_yaw': x[0][19],
        'gate3_x': x[0][20],
        'gate3_y': x[0][21],
        'gate3_z': x[0][22],
        'gate3_yaw': x[0][23],
        'gate4_x': x[0][24],
        'gate4_y': x[0][25],
        'gate4_z': x[0][26],
        'gate4_yaw': x[0][27],
        'gate1_in_range': x[0][28],
        'gate2_in_range': x[0][29],
        'gate3_in_range': x[0][30],
        'gate4_in_range': x[0][31],
        'obstacle1_x': x[0][32],
        'obstacle1_y': x[0][33],
        'obstacle1_z': x[0][34],
        'obstacle2_x': x[0][35],
        'obstacle2_y': x[0][36],
        'obstacle2_z': x[0][37],
        'obstacle3_x': x[0][38],
        'obstacle3_y': x[0][39],
        'obstacle3_z': x[0][40],
        'obstacle4_x': x[0][41],
        'obstacle4_y': x[0][42],
        'obstacle4_z': x[0][43],
        'obstacle1_in_range': x[0][44],
        'obstacle2_in_range': x[0][45],
        'obstacle3_in_range': x[0][46],
        'obstacle4_in_range': x[0][47],
        'gate_id': x[0][48]
    }

    if print_flag == True:
        print("Drone Position: ", observations['drone_pos'])
        print("Drone Roll, Pitch, Yaw: ", observations['drone_rpy'])
        print("Drone Velocity (X, Y, Z): ", observations['drone_vxyz'])
        print("Drone Angular Velocity (Roll, Pitch, Yaw): ", observations['drone_vrpy'])
        print("Gates (X, Y, Z, Yaw): ", observations['gates_xyz_yaw'])
        print("Gates in Range: ", observations['gates_in_range'])
        print("Obstacles (X, Y, Z): ", observations['obstacles_xyz'])
        print("Obstacles in Range: ", observations['obstacles_in_range'])
        print("Gate ID: ", observations['gate_id'])

    return observations

def load_configs(config_path):
            """
            Load configurations from a YAML file.

            Args:
                config_path (str): Path to the YAML file.

            Returns:
                dict: A dictionary containing the loaded configurations.
            """
            config_df = pd.DataFrame()
            with open(config_path, 'r') as file:
                configs = yaml.safe_load(file)
            return configs

def get_initial_observation(configs):
        """
        Get the initial observation data.

        Returns:
            ndarray: The initial observation data.
        """
        init_gates = configs["quadrotor_config"]["gates"]
        init_obstacles = configs["quadrotor_config"]["obstacles"]
        init_drone = configs["quadrotor_config"]["init_state"]
        return init_gates, init_obstacles, init_drone


def _generate_ref_waipoints(config_path = "config/getting_started.yaml"):

        configs = load_configs(config_path)
        init_gates, init_obstacles, init_drone = get_initial_observation(configs)
        #print(configs)
        x_reference = configs["quadrotor_config"]["task_info"]["stabilization_goal"]
        waypoints = []
        waypoints.append([init_drone["init_x"], init_drone["init_y"], 0.3])
        gates = init_gates
        z_low = 0.525
        z_high = 1.0
        waypoints.append([1, 0, z_low])
        waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
        waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
        waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.7,
                (gates[0][1] + gates[1][1]) / 2 - 0.3,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.5,
                (gates[0][1] + gates[1][1]) / 2 - 0.6,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
        waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
        waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])
        waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
        waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
        waypoints.append(
            [
                x_reference[0],
                x_reference[1],
                x_reference[2],
            ]
        )
        waypoints.append(
            [
                x_reference[0],
                x_reference[1] - 0.2,
                x_reference[2],
            ]
        )
        waypoints = np.array(waypoints)

        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        duration = 12
        t = np.linspace(0, 1, int(duration * configs["quadrotor_config"]["ctrl_freq"]))
        ref_x, ref_y, ref_z = interpolate.splev(t, tck)

        takeoff_duration = 1  # seconds
        takeoff_steps = int(takeoff_duration * configs["quadrotor_config"]["ctrl_freq"])
        takeoff_z = np.linspace(init_drone["init_z"], 0.3, takeoff_steps)

        waypoints = np.append([init_drone["init_x"], init_drone["init_y"], init_drone["init_z"]], waypoints)
        waypoints = np.append([init_drone["init_x"], init_drone["init_y"], 0.3], waypoints)

        ref_x = np.concatenate((np.full(takeoff_steps, init_drone["init_x"]), ref_x))
        ref_y = np.concatenate((np.full(takeoff_steps, init_drone["init_y"]), ref_y))
        ref_z = np.concatenate((takeoff_z, ref_z))

        #plot reference trajectory in 3D
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(ref_x, ref_y, ref_z)    
            # Plot start and end point
            ax.scatter(ref_x[0], ref_y[0], ref_z[0], c='green', label='Start')
            ax.scatter(ref_x[-1], ref_y[-1], ref_z[-1], c='red', label='End')

            # Plot the initial position
            ax.scatter(init_drone["init_x"], init_drone["init_y"], 0.3, c='orange', label='Initial Position')

            # Plot vertical lines for gates
            for gate_num, gate in enumerate(gates):
                x, y, z, _, _, _, _ = gate
                ax.plot([x, x], [y, y], [0, 1], label=f'Gate {gate_num+1}')

            # Plot vertical lines for obstacles
            #for obstacle in self.OBSTACLES:
            #    x, y, z, _, _, _ = obstacle
            #   ax.plot([x, x], [y, y], [0, 1], color='red', label='Obstacle')
            ax.legend()
            plt.show()

        assert max(ref_z) < 2.5, "Drone must stay below the ceiling"
        return ref_x, ref_y, ref_z, waypoints