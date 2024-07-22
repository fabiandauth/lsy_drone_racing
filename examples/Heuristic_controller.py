"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################
        self.VERBOSE = False
        self.initial_info = initial_info

        # initial information
        self.start = initial_obs[0:4]
        self.goal = [0, -1.5, 0.5]
        self.gates = list(
            [initial_obs[12:12 + 4], initial_obs[16:16 + 4], initial_obs[20:20 + 4], initial_obs[24:24 + 4]])
        self.obstacles = list(
            [initial_obs[32:32 + 3], initial_obs[35:35 + 3], initial_obs[38:38 + 3], initial_obs[41:41 + 3]])
        self.updated_gates = [0, 0, 0, 0]
        self.updated_obstacles = [0, 0, 0, 0]
        self.passed_gates = [0, 0, 0, 0]

        # calculate the trajectory
        waypoints = self._regen_waypoints(self.gates, self.obstacles, self.start[0:3], self.goal)
        self._recalc_trajectory(waypoints)

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.acc_x, self.acc_y, self.acc_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        self.step = 0
        self.last_step = 0

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)

        # check if gate has been passed
        pos = obs[0:3]
        self._check_if_gate_passed(pos)

        # check for gate or obstacle update
        gate_updated = self._update_gate_parameter(obs)
        obstacle_update = self._update_obstacle_parameter(obs)

        # process obstacle and gate updates
        if gate_updated:
            waypoints = self._regen_waypoints(self.gates, self.obstacles, pos, self.goal)
            self._recalc_trajectory(waypoints)
            self.last_step = 14
        if obstacle_update:
            if self._find_next_gate() < 1:
                index, obstacle = self._check_collision([self.obstacles[0]], self.acc_x, self.acc_y, self.acc_z)
                if index is not None:
                    self.waypoints[index, 0] = self.waypoints[index, 0] - 0.23
                    self._recalc_trajectory(self.waypoints)
                    self.last_step = self.last_step - 10


        # get next waypoint through speed control
        next_gate_index = self._find_next_gate()
        if next_gate_index == None:
            step = self.last_step + 3
        else:
            if next_gate_index == 0:
                step = self._speed_control(pos, self.gates[next_gate_index], self.start, self.last_step)
            else:
                step = self._speed_control(pos, self.gates[next_gate_index], self.gates[next_gate_index - 1], self.last_step)
        self.last_step = step

        # send Fullstate Command to Mellinger Controller
        if step < len(self.acc_x):
            target_pos = np.array([self.acc_x[step], self.acc_y[step], self.acc_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            command_type = Command.FULLSTATE
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
        # Notify set point stop has to be called every time we transition from low-level
        # commands to high-level ones. Prepares for landing
        elif step >= len(self.acc_x) and not self._setpoint_land:
            command_type = Command.NOTIFYSETPOINTSTOP
            args = []
            self._setpoint_land = True
        elif step >= len(self.acc_x) and not self._land:
            command_type = Command.LAND
            args = [0.0, 2.0]  # Height, duration
            self._land = True  # Send landing command only once
        elif self._land:
            command_type = Command.FINISHED
            args = []
        else:
            command_type = Command.NONE
            args = []

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)


    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer



    def _find_next_gate(self):
        """ Helper Function to find the next gate that has not been passed yet

                        Returns
                        ------
                        gate index: integer
                            gate index of the next gate in self.gates (None if all gates passed)
        """
        for i in range(len(self.passed_gates)):
            if self.passed_gates[i] == 0:
                return i
        # all gates are passed
        return None

    def _check_if_gate_passed(self, pos):
        """ Updates the list of passed gates if the position of the drone is within 0.13 of the gate middle

                        Parameters
                        ----------
                        pos : list
                            current position of the drone

        """
        for gate in range(0, len(self.gates[0])):
            if np.allclose(pos, self.gates[gate][0:3], atol=0.13):
                self.passed_gates[gate] = 1

    def _update_gate_parameter(self, obs):
        """ Updates the gate parameters if new observations are received

                        Parameters
                        ----------
                        obs : list
                            list of observations from the environment

                        Returns
                        ------
                        update: Bool
                            True : if an update to the gate positions has been made
                            False : if no update has been made
        """
        list_index = 0
        update = False
        for i in range(12, 25, 4):
            if not np.array_equal(obs[i:i+4], self.gates[list_index]) and self.updated_gates[list_index] == 0:
                self.gates[list_index] = obs[i : i + 4]
                self.updated_gates[list_index] = 1
                update = True

            list_index += 1
        return update


    def _speed_control(self, pos, next_gate, last_gate, last_waypoint):
        """ Updates the gate parameters if new observations are received

                        Parameters
                        ----------
                        pos : list[3]
                            current drone position
                        next_gate : list
                            position of the next gate
                        last_gate : list
                            position of the last gate
                        last_waypoint : list[3]
                            last waypoint given to the controller

                        Returns
                        ------
                        next_waypoint : list[3]
                            next_waypoint that can be given to the controller
        """
        distance_next_gate = np.linalg.norm(next_gate[0:3] - pos[0:3])
        distance_last_gate = np.linalg.norm(last_gate[0:3] - pos[0:3])
        min_speed = 1.51

        # adjust distance of next waypoint according to progress between gates
        if distance_next_gate > distance_last_gate:
            if distance_last_gate > 0.1:
                next_waypoint = round(last_waypoint + min_speed + 2)
            elif distance_last_gate > 0.3:
                next_waypoint = round(last_waypoint + min_speed + 2)
            else:
                next_waypoint = round(last_waypoint + min_speed + np.exp(1.9 * distance_last_gate) + 0.4)
        else:

            next_waypoint = round(last_waypoint + min_speed + 1)
        return next_waypoint

    def _update_obstacle_parameter(self, obs):
        """ Updates the obstacle parameters if new obstacle positions are received

                        Parameters
                        ----------
                        obs : list
                            list of observations from the environment

                        Returns
                        ------
                        update: Bool
                            True : if an update to the obstacle (self.obstacles) positions has been made
                            False : if no update has been made
        """
        list_index = 0
        update = False
        for i in range(32, 42, 3):
            if not np.array_equal(obs[i:i + 2], self.obstacles[list_index][0:2]) and self.updated_obstacles[list_index] == 0:
                self.obstacles[list_index] = obs[i : i + 3]
                self.obstacles[list_index][2] = 0.5
                self.updated_obstacles[list_index] = 1
                update = True
            list_index += 1
        return update     #updated obstacle parameter route recalculation necessary

    def _resolve_collision(self,
                           obstacles,
                           gate_pos,
                           direction,
                           lengths=[0.25, 0.2],
                           allowed_rot=[0, np.pi / 5, -np.pi / 5]):
        """ Finds the waypoints that maximizes the distance to a close obstacle based on
            a set of length and rotational offsets

                        Parameters
                        ----------
                        obstacles : 2D-list
                            list of observations from the environment
                        gate_pos : list
                            position of the gate where a collision has to be resolved
                        direction : int
                            -1 : back side of the gate
                            1  : front side of the gate
                        lengths : list (optional)
                            array of allowed distances from the gate middle point
                        allowed_rot : list (optinal)
                            array of allow rotational offsets from the gate normal in x-y

                        Returns
                        ------
                        return_pos: list[3]
                            resulting position as list of [x,y,z]
        """
        rot = gate_pos[3]
        best_distance = 0
        return_pos = None
        for length in lengths:
            for offset in allowed_rot:

                delta_x_ = np.cos(rot + np.pi / 2 + offset)
                delta_y_ = np.sin(rot + np.pi / 2 + offset)

                goal_pos = [gate_pos[0] + direction * delta_x_ * length, gate_pos[1] + direction * delta_y_ * length,
                            gate_pos[2]]
                for obstacle in obstacles:
                    distance = np.linalg.norm(np.array(goal_pos[0:2]) - np.array(obstacle[0:2]))
                    if distance > best_distance and distance < 0.7:
                        best_distance = distance
                        return_pos = goal_pos
        return return_pos

    def _recalc_trajectory(self, waypoints):
        """ Recalculates the spline of the given waypoints and
            asserts the resulting points on the spline to self.ref(x,y,z) and self.acc_(x,y,z)

                                Parameters
                                ----------
                                waypoints : list
                                    list of observations from the environment
                                iteration : list
                                    iteration paramter of the last trajectory
        """
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.05)
        self.waypoints = waypoints
        t_accurate = np.linspace(0, 1, int(1000 / (self._find_next_gate() + 1)))
        self.acc_x, self.acc_y, self.acc_z = interpolate.splev(t_accurate, tck)
        assert max(self.acc_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(self.initial_info, self.waypoints, self.acc_x, self.acc_y, self.acc_z)

    def _check_collision(self, obstacles, x, y, z):
        """ Checks if a spline point (self.acc(x,y,z)) is within 0.4 m of an obstacle
            and returns the closest waypoint together with the obstacle position

                            Parameters
                            ----------
                            obstacles : 2D-array
                                list of obstacles
                            x : double
                                x parameters of the spline points
                            y : double
                                y parameters of the spline points
                            z : double
                                z parameters of the spline points

                            Returns
                            ------
                            collision_index: integer
                                waypoint index closest to the collision
                            obstacle : list
                                obstacle position responsible for the collision
            """
        for obstacle in obstacles:
            for i in range(len(self.acc_x)):
                point = [x[i], y[i], z[i]]
                if np.linalg.norm(np.array(point[0:2]) - np.array(obstacle[0:2])) < 0.4:
                    if point[2] < 1.05:
                        collision_index = self._get_collision_waypoint(obstacle)
                        return collision_index, obstacle
        return None, None

    def _get_collision_waypoint(self, obstacle):
        """ Helper Function to return the collision waypoint

                                Parameters
                                ----------
                                obstacle : 2D-array
                                    list of obstacles

                                Returns
                                ------
                                collision_index
        """
        distances = np.linalg.norm(self.waypoints - obstacle, axis=1)
        return np.argmin(distances)


    def _regen_waypoints(self, gates, obstacles, pos, goal):
        """ Checks if a spline point (self.acc(x,y,z)) is within 0.4 m of an obstacle
            and returns the closest waypoint together with the obstacle position

                                Parameters
                                ----------
                                gates: 2D-array
                                    list of gates
                                obstacles : 2D-array
                                    list of obstacles
                                pos : list[x, y, z]
                                    current drone position
                                goal : list[x, y, z]
                                    drone position
        """
        next_gate_index = self._find_next_gate()
        z_low = self.initial_info["gate_dimensions"]["low"]["height"]
        z_high = self.initial_info["gate_dimensions"]["tall"]["height"]

        waypoints = []
        waypoints.append(pos)

        # append the waypoints according to gate progress
        if next_gate_index < 1 and np.linalg.norm(pos[0:3] - self.start[0:3]) < 0.5:
            waypoints.append([self.initial_obs[0], self.initial_obs[1], 0.1])
            waypoints.append([self.initial_obs[0], self.initial_obs[1], 0.3])
            waypoints.append([0.8, -0.5, z_low])

        if next_gate_index< 1:
            waypoints.append(self._resolve_collision(obstacles,gates[0], -1, allowed_rot=[0], lengths=[0.2, 0.1]))
            waypoints.append([gates[0][0], gates[0][1], gates[0][2]])
            waypoints.append(self._resolve_collision(obstacles, gates[0], 1))
            waypoints.append(
                [
                    (gates[0][0] + gates[1][0]) / 2 - 0.7,
                    (gates[0][1] + gates[1][1]) / 2 - 0.45,
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
        if next_gate_index < 2:
            if np.linalg.norm(pos - gates[0][0:3]) < np.linalg.norm(pos - gates[1][0:3]):
                waypoints.append(
                    [
                        (gates[0][0] + gates[1][0]) / 2 - 0.7,
                        (gates[0][1] + gates[1][1]) / 2 - 0.45,
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
            waypoints.append([gates[1][0], gates[1][1], gates[1][2]])
            waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
            waypoints.append([
                        (gates[1][0] + gates[2][0]) / 2,
                        (gates[1][1] + gates[2][1]) / 2,
                        (z_low + z_high) / 2,
                    ])

        if next_gate_index < 3:
            waypoints.append(self._resolve_collision(obstacles, gates[2], -1))
            waypoints.append([gates[2][0], gates[2][1], gates[2][2]])
            waypoints.append(self._resolve_collision(obstacles, gates[2], 1, lengths=[0.4, 0.45], allowed_rot= [0, np.pi/6, -np.pi/6]))
            point = self._resolve_collision(obstacles, gates[2], 1, lengths=[0.4, 0.45], allowed_rot=[0, np.pi/6, -np.pi/6])
            point[2] = z_high + 0.1
            point[1] = point[1] + 0.3
            waypoints.append(point)

        if next_gate_index < 4:
            waypoints.append([gates[3][0], gates[3][1] + 0.3, z_high+ 0.2])
            waypoints.append([gates[3][0], gates[3][1], gates[3][2]])
            waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.05])


        waypoints.append(
            [
                self.initial_info["x_reference"][0],
                self.initial_info["x_reference"][2],
                self.initial_info["x_reference"][4],
            ]
        )
        waypoints.append(
            [
                self.initial_info["x_reference"][0],
                self.initial_info["x_reference"][2] - 0.2,
                self.initial_info["x_reference"][4],
            ]
        )
        waypoints = np.array(waypoints)
        return waypoints