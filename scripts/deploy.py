#!/usr/bin/env python
"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.yaml>

"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import fire
import numpy as np
import rospy
import yaml
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from lsy_drone_racing.command import Command, apply_command
from lsy_drone_racing.import_utils import get_ros_package_path, pycrazyswarm
from lsy_drone_racing.utils import check_gate_pass, load_controller
from lsy_drone_racing.vicon import Vicon

logger = logging.getLogger(__name__)


def create_init_info(
    env_info: dict,
    gate_poses: list,
    obstacle_poses: list,
    constraint_values: list,
    x_reference: list,
) -> dict:
    """Create the initial information dictionary for the controller.

    Args:
        env_info: The environment information dictionary.
        gate_poses: The list of gate poses.
        obstacle_poses: The list of obstacle poses.
        constraint_values: The values of the environment constraints evaluated at the start state.
        x_reference: The reference state for the controller to stabilize the drone after the race.
    """
    gate_dimensions_low = {"shape": "square", "height": 0.525, "edge": 0.45}
    gate_dimensions_tall = {"shape": "square", "height": 1.0, "edge": 0.45}
    gate_dimensions = {"low": gate_dimensions_low, "tall": gate_dimensions_tall}
    obstacle_dimensions = {"shape": "cylinder", "height": 1.05, "radius": 0.05}
    physical_params = {
        "quadrotor_mass": 0.03454,
        "quadrotor_ixx_inertia": 1.4e-05,
        "quadrotor_iyy_inertia": 1.4e-05,
        "quadrotor_izz_inertia": 2.17e-05,
    }
    init_info = {
        "symbolic_model": env_info["symbolic_model"],
        "nominal_physical_parameters": physical_params,
        "x_reference": [x_reference[0], 0.0, x_reference[1], 0.0, x_reference[2], 0.0] + [0.0] * 6,
        "u_reference": [0.084623, 0.084623, 0.084623, 0.084623],
        "symbolic_constraints": env_info["symbolic_constraints"],
        "ctrl_timestep": 1 / 30,
        "ctrl_freq": 30,
        "episode_len_sec": 33,
        "quadrotor_kf": 3.16e-10,
        "quadrotor_km": 7.94e-12,
        "gate_dimensions": gate_dimensions,
        "obstacle_dimensions": obstacle_dimensions,
        "nominal_gates_pos_and_type": gate_poses,
        "nominal_obstacles_pos": obstacle_poses,
        "initial_state_randomization": env_info["initial_state_randomization"],
        "inertial_prop_randomization": env_info["inertial_prop_randomization"],
        "gates_and_obs_randomization": env_info["gates_and_obs_randomization"],
        "disturbances": env_info["disturbances"],
        "urdf_dir": None,
        "pyb_client": None,
        "constraint_values": constraint_values,
    }
    info.update(get_info(env, vicon, 0, config))
    return info


def sync_env(env: Quadrotor, vicon: Vicon):
    """Synchronize the internal env state with the real observations from vicon.

    Args:
        env: The Quadrotor environment.
        vicon: Vicon interface.
    """
    p, r = vicon.pos[vicon.drone_name], vicon.rpy[vicon.drone_name]
    env.state = [p[0], 0, p[1], 0, p[2], 0, r[0], r[1], r[2], 0, 0, 0]
    constraint_values = env.constraints.get_values(env, only_state=True)
    # IROS 2022 - Constrain violation flag for reward.
    env.cnstr_violation = env.constraints.is_violated(env, c_value=constraint_values)


def get_observation(info: dict, vicon: Vicon) -> np.ndarray:
    """Get the current observation from Vicon data.

    Args:
        info: The info dictionary corresponding to the current observation.
        vicon: Vicon interface.

    Returns:
        An observation from real data that matches the simulated observation space.
    """
    pos, rpy = vicon.pos[vicon.drone_name], vicon.rpy[vicon.drone_name]
    vel = vicon.vel[vicon.drone_name]
    ang_vel = vicon.ang_vel[vicon.drone_name]
    obs = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], *rpy, *ang_vel])
    return DroneRacingWrapper.observation_transform(obs, info)


def get_info(env: Quadrotor, vicon: Vicon, gate_id: int, config: Munch) -> dict:
    """Get the current environment info from Vicon data.

    Args:
        env: Quadrotor environment.
        vicon: Vicon interface.
        gate_id: Target gate ID.
        config: Competition configuration.

    Returns:
        An observation from real data that matches the simulated observation space.
    """
    gates = np.array(config.quadrotor_config.gates)
    # Infer gate z position based on type
    gates[:, 2] = np.where(gates[:, 6] == 1, Z_LOW, Z_HIGH)
    obstacles_pose = np.array(config.quadrotor_config.obstacles)
    obstacles_pose[:, 2] = ObstacleDesc.height
    drone_pos = vicon.pos[vicon.drone_name][:2]
    gates_in_range = np.linalg.norm(gates[:, :2] - drone_pos, axis=1) < SENSOR_RANGE
    obstacles_in_range = np.linalg.norm(obstacles_pose[:, :2] - drone_pos, axis=1) < SENSOR_RANGE

    # Update gates and obstacles that are closer than SENSOR_RANGE
    for i, in_range in enumerate(gates_in_range):
        if in_range:  # Overwrite both position and orientation
            gates[i, :2] = vicon.pos[f"gate{i+1}"][:2]
            gates[i, 3] = vicon.rpy[f"gate{i+1}"][2]
    for i, in_range in enumerate(obstacles_in_range):
        if in_range:  # Obstacles are symmetric, don't use orientation
            obstacles_pose[i, :2] = vicon.pos[f"obstacle{i+1}"][:2]

    info = {
        "mse": np.sum(((env.state - env.X_GOAL) * env.info_mse_metric_state_weight) ** 2),
        "collision": (None, False),  # Leave always False for deploy
        "gates_pose": gates[:, :6],
        "obstacles_pose": obstacles_pose,
        "gates_in_range": gates_in_range,
        "obstacles_in_range": obstacles_in_range,
        "gates_type": gates[:, 6],
        "current_gate_id": gate_id,
        "at_goal_position": False,  # Leave always False for deployment
        "task_completed": False,  # Leave always False for deployment
        "drone_pose": np.concatenate(vicon.pose(vicon.drone_name)),
        "constraint_values": env.constraints.get_values(env, only_state=True),
        "constraint_violation": 1 if env.cnstr_violation else 0,
    }
    return info
    return init_info


def main(config: str = "config/getting_started.yaml", controller: str = "examples/controller.py"):
    """Deployment script to run the controller on the real drone."""
    start_time = time.time()

    # Load the controller and initialize the crazyfly interface
    Controller = load_controller(Path(controller))
    crazyswarm_config_path = get_ros_package_path("crazyswarm") / "launch/crazyflies.yaml"
    # pycrazyswarm expects strings, not Path objects, so we need to convert it first
    swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
    time_helper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    gate_names = [f"gate{i}" for i in range(1, len(config.quadrotor_config.gates) + 1)]
    obstacle_names = [f"obstacle{i}" for i in range(1, len(config.quadrotor_config.obstacles) + 1)]
    vicon = Vicon(track_names=gate_names + obstacle_names, timeout=1.0)

    config_path = Path(config).resolve()
    assert config_path.is_file(), "Config file does not exist!"
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    config_factory = ConfigFactory()
    config_factory.base_dict = config
    config = config_factory.merge()

    # Check if the real drone position matches the settings
    tol = 0.1
    init_state = config.quadrotor_config.init_state
    drone_pos = np.array([init_state[key] for key in ("init_x", "init_y", "init_z")])
    if d := np.linalg.norm(drone_pos - vicon.pos["cf"]) > tol:
        raise RuntimeError(
            (
                f"Distance between drone and starting position too great ({d:.2f}m)"
                f"Position is {vicon.pos['cf']}, should be {drone_pos}"
            )
        )

    # TODO: Replace with autodetection of gate and obstacle positions
    # TODO: Change obstacle and gate definitions to freely adjust the height
    gate_poses = config.quadrotor_config.gates
    for gate in gate_poses:
        if gate[3] != 0 or gate[4] != 0:
            raise ValueError("Gates can't have roll or pitch!")
    obstacle_poses = config.quadrotor_config.obstacles

    # Create a safe-control-gym environment from which to take the symbolic models
    config.quadrotor_config["ctrl_freq"] = 500
    env = make("quadrotor", **config.quadrotor_config)
    _, env_info = env.reset()

    # Override environment state and evaluate constraints
    drone_pos, drone_rot = vicon.pos[vicon.drone_name], vicon.rpy[vicon.drone_name]
    drone_pos_and_vel = [drone_pos[0], 0, drone_pos[1], 0, drone_pos[2], 0]
    drone_rot_and_agl_vel = [drone_rot[0], drone_rot[1], drone_rot[2], 0, 0, 0]
    env.state = drone_pos_and_vel + drone_rot_and_agl_vel
    constraint_values = env.constraints.get_values(env, only_state=True)
    x_reference = config.quadrotor_config.task_info.stabilization_goal

    init_info = create_init_info(
        env_info, gate_poses, obstacle_poses, constraint_values, x_reference
    )

    CTRL_FREQ = init_info["ctrl_freq"]

    # Create controller
    vicon_obs = drone_pos_and_vel + drone_rot_and_agl_vel
    ctrl = Controller(vicon_obs, init_info, True)

    # Helper parameters
    target_gate_id = 0  # Initial gate.
    log_cmd = []  # Log commands as [current time, ros time, command type, args]
    last_drone_pos = vicon.pos["cf"].copy()  # Gate crossing helper
    completed = False
    print(f"Setup time: {time.time() - start_time:.3}s")

    try:
        # Run the main control loop
        start_time = time.time()
        total_time = None
        while not time_helper.isShutdown():
            curr_time = time.time() - start_time

            # Override environment state and evaluate constraints
            p, r = vicon.pos["cf"], vicon.rpy["cf"]
            env.state = [p[0], 0, p[1], 0, p[2], 0, r[0], r[1], r[2], 0, 0, 0]
            state_error = (env.state - env.X_GOAL) * env.info_mse_metric_state_weight
            constraint_values = env.constraints.get_values(env, only_state=True)
            # IROS 2022 - Constrain violation flag for reward.
            env.cnstr_violation = env.constraints.is_violated(env, c_value=constraint_values)
            cnstr_num = 1 if env.cnstr_violation else 0

            p = vicon.pos["cf"]
            # This only looks at the x-y plane, could be improved
            # TODO: Replace with 3D distance once gate poses are given with height
            gate_dist = np.sqrt(np.sum((p[0:2] - vicon.pos[gate_names[target_gate_id]][0:2]) ** 2))
            if gate_dist < 0.45:
                current_target_gate_pos = vicon.pos[gate_names[target_gate_id]]
            else:
                current_target_gate_pos = gate_poses[target_gate_id][0:6]
            info = {
                "mse": np.sum(state_error**2),
                "collision": (None, False),  # Leave always false in sim2real
                "current_target_gate_id": target_gate_id,
                "current_target_gate_in_range": gate_dist < 0.45,
                "current_target_gate_pos": current_target_gate_pos,
                "current_target_gate_type": gate_poses[target_gate_id][6],
                "at_goal_position": False,  # Leave always false in sim2real
                "task_completed": False,  # Leave always false in sim2real
                "constraint_values": constraint_values,
                "constraint_violation": cnstr_num,
            }

            # Check if the drone has passed the current gate
            if check_gate_pass(
                gate_poses[target_gate_id], vicon.pos[vicon.drone_name], last_drone_pos
            ):
                target_gate_id += 1
                print(f"Gate {target_gate_id} passed in {curr_time:.4}s")
            last_drone_pos = vicon.pos[vicon.drone_name].copy()

            if target_gate_id == len(gate_poses):  # Reached the end
                target_gate_id = -1
                total_time = time.time() - start_time

            # Get the latest vicon observation and call the controller
            p = vicon.pos[vicon.drone_name]
            drone_pos_and_vel = [p[0], 0, p[1], 0, p[2], 0]
            r = vicon.rpy[vicon.drone_name]
            drone_rot_and_agl_vel = [r[0], r[1], r[2], 0, 0, 0]
            vicon_obs = drone_pos_and_vel + drone_rot_and_agl_vel
            # In sim2real: Reward always 0, done always false
            command_type, args = ctrl.compute_control(curr_time, vicon_obs, 0, False, info)
            log_cmd.append([curr_time, rospy.get_time(), command_type, args])  # Save for logging

            apply_command(cf, command_type, args)  # Send the command to the drone controller
            time_helper.sleepForRate(CTRL_FREQ)

            if command_type == Command.FINISHED or completed:
                break

        logger.info(f"Total time: {total_time:.3f}s" if total_time else "Task not completed")
        # Save the commands for logging
        save_dir = Path(__file__).resolve().parents[1] / "logs"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "log.pkl", "wb") as f:
            pickle.dump(log_cmd, f)
    finally:
        apply_command(cf, Command.NOTIFYSETPOINTSTOP, [])
        apply_command(cf, Command.LAND, [0.0, 2.0])  # Args are height and duration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
