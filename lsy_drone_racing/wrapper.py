"""Wrapper to make the environment compatible with the gymnasium API.

The drone simulator does not conform to the gymnasium API, which is used by most RL frameworks. This
wrapper can be used as a translation layer between these modules and the simulation.

RL environments are expected to have a uniform action interface. However, the Crazyflie commands are
highly heterogeneous. Users have to make a discrete action choice, each of which comes with varying
additional arguments. Such an interface is impractical for most standard RL algorithms. Therefore,
we restrict the action space to only include FullStateCommands.

We also include the gate pose and range in the observation space. This information is usually
available in the info dict, but since it is vital information for the agent, we include it directly
in the observation space.

Warning:
    The RL wrapper uses a reduced action space and a transformed observation space!
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper

from lsy_drone_racing.rotations import map2pi

logger = logging.getLogger(__name__)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, terminate_on_lap: bool = True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.env = env
        # Unwrapped attribute is required for the gymnasium API. Some packages like stable-baselines
        # use it to check if the environment is unique. Therefore, we cannot use None, as None is
        # None returns True and falsely indicates that the environment is not unique. Lists have
        # unique id()s, so we use lists as a dummy instead.
        self.env.unwrapped = []
        self.env.render_mode = None

        # Gymnasium env required attributes
        # Action space:
        # [dx, dy, dz, dyaw]
        # dx, dy, dz)   Desired position change of the drone relative to its current location.
        # dyaw)         Desired yaw angle change of the drone relative to its current orientation.
        # dx, dy, dz remain unscaled, i.e. they represent a desired position change in meters. dyaw
        # is scaled up to [-pi, pi]. The yaw value of 1, e.g., corresponds to pi radians.
        # Example:
        #       An action of [0, 0, 0, 0] corresponds to no change
        #       An action of [1, 0, -1, 1] at pose [0.5, 1.0, 2.3, 0] corresponds to a desired
        #       pose of [1.5, 1, 1.3, np.pi].
        self.action_scale = np.array([1, 1, 1, 0])
        self.action_space = Box(-0.5, 0.5, shape=(4,), dtype=np.float32)

        # Observation space:
        # [drone_xyz, drone_rpy, drone_vxyz, drone vrpy, gates_xyz_yaw, gates_in_range,
        # obstacles_xyz, obstacles_in_range, gate_id]
        # drone_xyz)  Drone position in meters.
        # drone_rpy)  Drone orientation in radians.
        # drone_vxyz)  Drone velocity in m/s.
        # drone_vrpy)  Drone angular velocity in rad/s.
        # gates_xyz_yaw)  The pose of the gates. Positions are in meters and yaw in radians. The
        #       length is dependent on the number of gates. Ordering is [x0, y0, z0, yaw0, x1,...].
        # gates_in_range)  A boolean array indicating if the drone is within the gates' range. The
        #       length is dependent on the number of gates.
        # obstacles_xyz)  The pose of the obstacles. Positions are in meters. The length is
        #       dependent on the number of obstacles. Ordering is [x0, y0, z0, x1,...].
        # obstacles_in_range)  A boolean array indicating if the drone is within the obstacles'
        #       range. The length is dependent on the number of obstacles.
        # gate_id)  The ID of the current target gate. -1 if the task is completed.
        #n_gates = env.env.NUM_GATES
        n_gates = 1
        n_obstacles = env.env.n_obstacles
        # Velocity limits are set to 10 m/s for the drone and 10 rad/s for the angular velocity.
        # While drones could go faster in theory, it's not safe in practice and we don't allow it in
        # sim either.
        drone_limits = [5, 5, 5, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10, 10]
        distance_limits = [10, 10]  # Distance to gate and obstacle
        gate_limits = [5, 5, 5, np.pi] #* n_gates + [1] * n_gates  # Gate poses and range mask
        obstacle_limits = [5, 5, 5] * n_obstacles  # Obstacle pos and range mask
        obs_limits = drone_limits + distance_limits + gate_limits + obstacle_limits + [n_gates]  # [1] for gate_id
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        #self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

                # drone_pos,  # 3
                # drone_rpy,  # 3
                # drone_vel,  # 3
                # drone_ang_vel,  # 3
                # dist_to_gate,   # 1
                # dist_to_obstacles, # 1
                # info["gates_pose"][current_gate_id, [0, 1, 2, 5]].flatten(), # 4
                # #info["gates_in_range"],
                # info["obstacles_pose"][:, :3].flatten(), # 3
                # #info["obstacles_in_range"],
                # [info["current_gate_id"]], # 1

        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config and helper flags
        self.terminate_on_lap = terminate_on_lap
        self._reset_required = False
        # The original firmware wrapper requires a sim time as input to the step function. This
        # breaks the gymnasium interface. Instead, we keep track of the sim time here. On each step,
        # it is incremented by the control time step. On env reset, it is reset to 0.
        self._sim_time = 0.0
        self._drone_pose = None
        # The firmware quadrotor env requires the rotor forces as input to the step function. These
        # are zero initially and updated by the step function. We automatically insert them to
        # ensure compatibility with the gymnasium interface.
        # TODO: It is not clear if the rotor forces are even used in the firmware env. Initial tests
        #       suggest otherwise.
        self._f_rotors = np.zeros(4)

    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: The random seed to use for the environment. Not used in this wrapper.
            options: Additional options to pass to the environment. Not used in this wrapper.

        Returns:
            The initial observation and info dict of the next episode.
        """
        self._reset_required = False
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        obs, info = self.env.reset()
        # Store obstacle height for observation expansion during env steps.
        obs = self.observation_transform(obs, info).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        assert not self._reset_required, "Environment must be reset before taking a step"
        if action not in self.action_space:
            # Wrapper has a reduced action space compared to the firmware env to make it compatible
            # with the gymnasium interface and popular RL libraries.
            raise InvalidAction(f"Invalid action: {action}")
        action = self._action_transform(action)
        assert action.shape[-1] == 4, "Action must have shape (..., 4)"
        # The firmware does not use the action input in the step function
        zeros = np.zeros(3)
        self.env.sendFullStateCmd(action[:3], zeros, zeros, 0, zeros, self._sim_time)
        # The firmware quadrotor env requires the sim time as input to the step function. It also
        # returns the desired rotor forces. Both modifications are not part of the gymnasium
        # interface. We automatically insert the sim time and reuse the last rotor forces.
        obs, reward, done, info, f_rotors = self.env.step(self._sim_time, action=self._f_rotors)
        self._f_rotors[:] = f_rotors
        # We set truncated to True if the task is completed but the drone has not yet passed the
        # final gate. We set terminated to True if the task is completed and the drone has passed
        # the final gate.
        terminated, truncated = False, False
        if info["task_completed"] and info["current_gate_id"] != -1:
            truncated = True
        elif self.terminate_on_lap and info["current_gate_id"] == -1:
            info["task_completed"] = True
            terminated = True
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            terminated = True
        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt
        obs = self.observation_transform(obs, info).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        if obs not in self.observation_space:
            terminated = True
            reward = -1
        self._reset_required = terminated or truncated
        return obs, reward, terminated, truncated, info

    def _action_transform(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        action = self._drone_pose + (action * self.action_scale)
        return action

    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"

    @staticmethod
    def observation_transform(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Transform the observation to include additional information.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_vel = obs[1:6:2]
        drone_rpy = obs[6:9]
        drone_ang_vel = obs[9:12]

        current_gate_id = info["current_gate_id"]
        if info["gates_pose"][current_gate_id, -1] == 0:
            current_gate = [info["gates_pose"][current_gate_id, 0], info["gates_pose"][current_gate_id, 1], 0.525]
        else:
            current_gate = [info["gates_pose"][current_gate_id, 0], info["gates_pose"][current_gate_id, 1], 1]

        dist_to_gate = np.linalg.norm(current_gate - drone_pos)

        try: 
            obstacles_in_range_index = np.where(info["obstacles_in_range"] == 1)[0]
            dist_to_obstacles = np.linalg.norm(info["obstacles_pose"][obstacles_in_range_index, :2] - drone_pos[:2])
        except:
            dist_to_obstacles = 10 # if no obstacles in range, set to 10
        obs = np.concatenate(
            [
                drone_pos,  # 3 !
                drone_rpy,  # 3
                drone_vel,  # 3
                drone_ang_vel,  # 3
                [dist_to_gate],   # 1 !
                [dist_to_obstacles], # 1
                info["gates_pose"][current_gate_id, [0, 1, 2, 5]].flatten(), # 4
                #info["gates_in_range"],
                info["obstacles_pose"][:, :3].flatten(), # 3
                #info["obstacles_in_range"],
                [info["current_gate_id"]], # 1
            ]
        )
        return obs

class RewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self._last_gate = None

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self._last_gate = info["current_gate_id"]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """
        drone_pos = obs[:3]
        drone_rpy = obs[3:6]
        drone_vel = obs[6:9]
        drone_ang_vel = obs[9:12]
        dist_to_gate = obs[12]
        dist_to_obstacle = obs[13]
        gates_pose = obs[14:18]
        obstacle_pose = obs[18:21]
        current_gate_id = obs[21]

        z_low = 0.525
        z_high = 1

        rew = 0

        rew += 1/(dist_to_gate+0.1)

        if dist_to_obstacle < 0.4:
            rew -= 1/(dist_to_obstacle+0.1)

        rew = rew + reward

        return rew
        

        # gate_id = info["current_gate_id"]
        # gate_reward = np.exp(-np.linalg.norm(info["gates_pose"][gate_id, :3] - obs[:3]))
        # if gate_id == self._last_gate:
        #     gate_passed_reward = 0 
        # else:
        #     gate_passed_reward = 0.1
        # if terminated and not info["task_completed"]: 
        #     crash_penality = -1 
        # else:
        #     crash_penality = 0
        
        # rew = gate_reward + crash_penality + gate_passed_reward
        # print(f"Reward from _compute_reward: {rew}")
        return rew
