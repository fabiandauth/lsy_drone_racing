from pathlib import Path
from stable_baselines3 import PPO
from train import create_race_env
from lsy_drone_racing.train_utils import process_observation, save_observations
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from mpl_toolkits.mplot3d import Axes3D
from lsy_drone_racing.wrapper import RewardWrapper
import lsy_drone_racing.train_utils as train_utils
import pybullet as p




def play_trained_model(model_path: str, config_path: str, gui: bool = False,episodes_path: str = "episodes/"):
    """Load a trained model and play it in the environment."""
    # Create environment
    env = DummyVecEnv([lambda: create_race_env(Path(config_path), gui=True, multiprocess=False)])
    # Load the trained model
    model = PPO.load(model_path, env=env)
    # Set the model's environment
    #model.set_env(env)
    # Play the model in the environment
    ref_x, ref_y, ref_z, waypoints = train_utils._generate_ref_waipoints(config_path)
    print(ref_x.shape)
    episodes = 5
    for i in range(episodes):
        obs_list = []
        action_list = []
        x = env.reset()
        process_observation(x, False)
        done = False
        ret = 0.
        episode_length = 0
        j = 0
        
        # Draw the trajectory
        step = int(ref_x.shape[0]/50)
        for i in range(step, ref_x.shape[0], step):
            p.addUserDebugLine(
                lineFromXYZ=[ref_x[i - step], ref_y[i - step], ref_z[i - step]],
                lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                lineColorRGB=[1, 0, 0]
            )
        while not done:
            action, *_ = model.predict(x)
            action_list.append(action)
            #print(action)
            x, r, done, info = env.step(action)
            ret += r
            episode_length += 1
            obs_list.append(process_observation(x, False))
            j += 1
            #p.addUserDebugPoints([[ref_x[j],ref_y[j], ref_z[j]]], pointSize=3, pointColorsRGB=[[1,0,0]])
        # Save the observations
        print(f"Episode {i}: Return: {ret}, Episode Length: {episode_length}")
        save_path = episodes_path + "episodes"
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_observations(obs_list, save_path, i)

        # plot actions in 3D
        import matplotlib.pyplot as plt

        # Extract x, y, z coordinates from the action_list
        x_coords = [action[0][0] for action in action_list]
        y_coords = [action[0][1] for action in action_list]
        z_coords = [action[0][2] for action in action_list]

        # Create a 3D plot
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x_coords, y_coords, z_coords)

            # Set labels and title˚
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Actions in 3D')
            # Show the plot
            plt.show()

    return ret, episode_length

if __name__ == '__main__':
    model_path = "trained_models/2024-06-12_21-37-35/model_12320000_steps.zip"
    episodes_path = os.path.dirname(model_path) + "/"
    config_path = "config/getting_started.yaml"
    ret, episode_length = play_trained_model(model_path, config_path, 100, episodes_path)
    print(f"Return: {ret}, Episode length: {episode_length}")
