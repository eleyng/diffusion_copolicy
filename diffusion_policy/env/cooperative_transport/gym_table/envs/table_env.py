import os

os.environ["SDL_AUDIODRIVER"] = "dsp"  # for training on cluster
import math
import pickle
import random
import time
from os import mkdir
from os.path import exists
from typing import Dict, List, Tuple, Union

import cv2 as cv
import gym
import numpy as np
import pygame
import torch
from gym import spaces

# from gym.envs.classic_control import rendering
from PIL import Image

from diffusion_policy.env.cooperative_transport.gym_table.envs.custom_rewards import custom_reward_function
from diffusion_policy.env.cooperative_transport.gym_table.envs.game_objects.game_objects import (
    Agent,
    Obstacle,
    Table,
    Target,
)
from diffusion_policy.env.cooperative_transport.gym_table.envs.utils import (
    BLACK,
    FPS,
    STATE_H,
    STATE_W,
    WINDOW_H,
    WINDOW_W,
    debug_print,
    load_cfg,
    rad,
    set_action_joystick,
    set_action_keyboard,
)

VERBOSE = False  # For debugging


def debug_print(*args):
    if not VERBOSE:
        return
    print(*args)


class TableEnv(gym.Env):
    """The table environment.

    This environment consists of two agents rigidly attached to a table.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def handle_kwargs(self, obs, control, map_config, ep):
        self.obs_type = obs  # type of observation space, rgb or discrete
        self.control_type = control  # type of control input, keyboard or joystick
        self.map_cfg = load_cfg(map_config)
        # Episode initiation
        self.ep = ep

    def init_pygame(self):
        """Initialize the environment."""
        if self.render_mode == "headless":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            print("Running with display.")
        pygame.init()
        self.screen = pygame.display.set_mode(([WINDOW_W, WINDOW_H]))
        self.viewer = None

    def init_data_paths(self, map_config, run_mode, run_name):
        map_name = map_config.split("/")[-1].split(".")[0]
        root_dir = os.path.join(os.path.dirname(__file__), "../../../")
        print("root_dir", root_dir)
        if not exists(os.path.join(root_dir, run_mode)):
            debug_print("Making base directories.")
            mkdir(os.path.join(root_dir, run_mode))
        if not exists(os.path.join(root_dir, run_mode, map_name)):
            mkdir(os.path.join(root_dir, run_mode, map_name))
        self.base_dirname = os.path.join(root_dir, run_mode, map_name, run_name)
        self.dirname = os.path.join(
            root_dir, run_mode, map_name, run_name, "trajectories"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        self.dirname_fluency = os.path.join(
            root_dir, run_mode, map_name, run_name, "fluency"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        self.dirname_vis = os.path.join(
            root_dir, run_mode, map_name, run_name, "figures"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        self.map_config_dir = os.path.join(
            root_dir, run_mode, map_name, run_name, "map_cfg"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        if not exists(self.base_dirname):
            mkdir(self.base_dirname)
            mkdir(self.dirname)
            mkdir(self.dirname_fluency)
            mkdir(self.dirname_vis)
            mkdir(self.map_config_dir)
        debug_print("Saving to directory: ", self.base_dirname)

        self.file_name = os.path.join(self.dirname, "ep_" + str(self.ep) + ".pkl")
        self.config_file_name = os.path.join(
            self.map_config_dir, "ep_" + str(self.ep) + ".npz"
        )
        self.file_name_fluency = os.path.join(
            self.dirname_fluency, "ep_" + str(self.ep) + ".npz"
        )
        if not os.path.exists(os.path.dirname(self.file_name_fluency)):
            os.makedirs(os.path.dirname(self.file_name_fluency))
        if not os.path.exists(os.path.dirname(self.file_name)):
            os.makedirs(os.path.dirname(self.file_name))
        if not os.path.exists(os.path.dirname(self.config_file_name)):
            os.makedirs(os.path.dirname(self.config_file_name))

        self.dirname_vis_ep = os.path.join(
            self.dirname_vis, "ep_" + str(self.ep) + "_images"
        )
        if not exists(self.dirname_vis_ep):
            debug_print("Making image directory: ", self.dirname_vis)
            mkdir(self.dirname_vis_ep)

        debug_print("Data saved location: ", self.file_name)

    def init_env(self, load_map=None):
        self.init_pygame()
        # Initialize the observation history
        self.obs_hist = np.zeros(self.state_dim * self.seq_length, dtype=np.float32)
        self.n = 2  # number of players
        # load from saved env config file
        if load_map is not None and self.run_mode == "eval":
            map_run = dict(np.load(load_map, allow_pickle=True))
            # table init pose
            table_cfg = [
                map_run["table"].item()["x"] / WINDOW_W,
                map_run["table"].item()["y"] / WINDOW_H,
                map_run["table"].item()["angle"],
            ]
            # table goal pose
            goal_cfg = [
                map_run["goal"].item()["goal"][0] / WINDOW_W,
                map_run["goal"].item()["goal"][1] / WINDOW_H,
            ]
            # table obstacles as encoding
            obs_lst_cfg = map_run["obstacles"].item()["obs_lst"]
            num_obs_cfg = map_run["obstacles"].item()["num_obstacles"]
        else:
            # randomly sample new table init position
            table_cfg = self.map_cfg["TABLE"][
                random.sample(range(0, len(self.map_cfg["TABLE"])), 1)[0]
            ]

        self.table = Table(
            x=table_cfg[0] * WINDOW_W,
            y=table_cfg[1] * WINDOW_H,
            angle=table_cfg[2],
        )
        self.config_params = {}
        self.table_params = {}
        self.table_params["x"] = self.table.x
        self.table_params["y"] = self.table.y
        self.table_params["angle"] = self.table.angle
        self.table_init = np.array([self.table.x, self.table.y, self.table.angle])
        debug_print("Table initial configuration: ", self.table_params)

        self.player_1 = Agent()
        self.player_2 = Agent()

        # RANDOM OBSTACLE CONFIG
        self.obs_dim = len(self.map_cfg["OBSTACLES"][0]["POSITIONS"])
        self.num_obstacles = np.random.choice(range(1, self.max_num_obstacles + 1), 1)[
            0
        ]
        self.visible_obs = 1  # float(args["vis"])

        # create obstacle
        self.obs_lst_idx = random.sample(
            range(0, len(self.map_cfg["OBSTACLES"][0]["POSITIONS"])), self.num_obstacles
        )

        if load_map is not None and self.run_mode == "eval":
            self.obs_lst_idx = obs_lst_cfg
            self.num_obstacles = num_obs_cfg

        if self.set_obs is not None:
            self.obs_lst_idx = self.set_obs  # [1]
            self.num_obstacles = 1

        self.obs_lst = [
            self.map_cfg["OBSTACLES"][0]["POSITIONS"][i] for i in self.obs_lst_idx
        ]

        # initialize obstacles & obstacle sprites
        self.obstacles = np.zeros((self.num_obstacles, 2))
        self.obs_sprite = []

        for i in range(len(self.obs_lst)):
            obs = np.array(
                [
                    self.obs_lst[i][0] * WINDOW_W,
                    self.obs_lst[i][1] * WINDOW_H,
                ]
            )
            self.obstacles[i] = obs
            self.obs_sprite.append(
                Obstacle(
                    self.obstacles[i], size=self.map_cfg["OBSTACLES"][0]["SIZES"][0]
                )
            )
        self.obs_params = {}
        self.obs_params["obstacles"] = self.obstacles
        self.obs_params["num_obstacles"] = self.num_obstacles
        self.obs_params["obs_lst"] = self.obs_lst_idx
        self.obs_params["obs_dim"] = self.obs_dim
        debug_print("Obstacle configuration: ", self.obs_params)

        # RANDOM GOAL CONFIG
        goal_rnd = self.map_cfg["GOAL"][
            random.sample(range(0, len(self.map_cfg["GOAL"])), 1)[0]
        ]
        if load_map is not None and self.run_mode == "eval":
            goal_rnd = goal_cfg

        debug_print("goal_rnd", goal_rnd)
        self.goal = np.array([goal_rnd[0] * WINDOW_W, goal_rnd[1] * WINDOW_H])
        self.goal_params = {}
        self.goal_params["goal"] = self.goal
        debug_print("Goal configuration: ", self.goal_params)

        # MAP CONFIG
        self.map_info = np.concatenate((self.table_init, self.goal))

        # SAVE CONFIGURATION
        self.config_params = {}
        self.config_params["table"] = self.table_params
        self.config_params["obstacles"] = self.obs_params
        self.config_params["goal"] = self.goal_params

        # find dist2goal, dist2obs
        direction = self.goal - np.array([self.table.x, self.table.y])
        self.dist2goal = np.linalg.norm(direction)
        self.avoid = np.array(self.obstacles) - np.array([self.table.x, self.table.y])
        self.dist2obs = np.linalg.norm(self.avoid, axis=1)
        self.wallpts = np.array(
            [[0, 0], [0, WINDOW_H], [WINDOW_W, WINDOW_H], [WINDOW_W, 0], [0, 0]]
        )
        self.target = Target(self.goal)
        self.sprite_list = pygame.sprite.Group()
        self.sprite_list.add(self.table)
        self.sprite_list.add(self.target)
        if self.visible_obs == 1:
            for i in range(self.num_obstacles):
                self.sprite_list.add(self.obs_sprite[i])
        elif self.visible_obs == 0.5:
            for i in range(self.num_obstacles):
                prob = random.randint(0, 1)
                if prob:
                    self.sprite_list.add(self.obs_sprite[i])
        self.done_list = pygame.sprite.Group()
        self.done_list.add(self.target)
        self.done_list.add(self.obs_sprite)
        self.done = False
        self.success = False

        # update metrics through reward
        table_state = np.expand_dims(
            np.array([self.table.x, self.table.y, self.table.angle]), axis=0
        )
        self.update_metrics(table_state)
        # reward = self.compute_reward(np.array([table_state]))

        if self.occupancy_grid:
            self.grid = self.make_occupancy_grid(WINDOW_H, WINDOW_W)
        else:
            self.grid = np.zeros((self.max_num_obstacles * 2))
            self.grid[: self.obstacles.flatten().shape[0]] = self.obstacles.flatten()
        self.prev_time = time.time()
        self.observation = self.get_state()
        self.full_observation = self.observation.cpu().detach().numpy()
        self.n_step = 0

        self.cap = self.player_1.cap  # TODO: match force cap in game_objects

    def init_rnd_seed_space(self):
        self.random_seed_space = spaces.Box(-1.0, 1.0, (2,))

    def init_action_space(self):
        # ----------------------------------------------- Action and Observation Spaces -------------------------------------------------------------
        if self.control_type == "keyboard":
            # define action space
            self.action_space = spaces.Discrete(25)
        elif self.control_type == "joystick":
            # continuous action space specified by two pairs of joystick axis
            action_space_low = np.array([-1.0, -1.0, -1.0, -1.0])
            action_space_high = np.array([1.0, 1.0, 1.0, 1.0])
            self.action_space = spaces.Box(
                action_space_low, action_space_high, dtype=np.float32
            )
        else:
            raise NotImplementedError("Unknown control type: %s" % self.control_type)

    def init_observation_space(self):
        # discrete observation space
        if self.obs_type == "discrete":
            if not self.occupancy_grid:
                obs_space_low = np.array(
                    [0.0, 0.0, -1.0, -1.0, -50.0, -50.0, -np.pi / 2]
                )
                obs_space_low = np.tile(obs_space_low, self.seq_length)
                map_dim_low = np.array(
                    [
                        0,
                        0,
                        -2 * np.pi,
                        0,
                        0,
                    ]
                )
                grid_dim_low = np.zeros(shape=(self.max_num_obstacles * 2,))
                self.obs_space_low = np.concatenate(
                    (obs_space_low, map_dim_low, grid_dim_low)
                ).astype(np.float32)

                obs_space_hi = np.array(
                    [WINDOW_W, WINDOW_H, 1.0, 1.0, 50.0, 50.0, np.pi / 2]
                )

                obs_space_hi = np.tile(obs_space_hi, self.seq_length)
                map_dim_hi = np.array(
                    [
                        WINDOW_W,
                        WINDOW_H,
                        2 * np.pi,
                        WINDOW_W,
                        WINDOW_H,
                    ]
                )
                grid_dim_hi = np.tile(
                    np.array([WINDOW_W, WINDOW_H]).flatten(), self.max_num_obstacles
                )
                self.obs_space_hi = np.concatenate(
                    (obs_space_hi, map_dim_hi, grid_dim_hi)
                ).astype(np.float32)

            else:
                self.map_dim_w = int(WINDOW_W / self.scale)
                self.map_dim_h = int(WINDOW_H / self.scale)
                self.occupancy_grid_dim = self.map_dim_w * self.map_dim_h

                obs_space_low = np.array(
                    [0.0, 0.0, -1.0, -1.0, -50.0, -50.0, -np.pi / 2]
                )
                obs_space_low = np.tile(obs_space_low, self.seq_length)
                map_dim_low = np.array(
                    [
                        0,
                        0,
                        -2 * np.pi,
                        0,
                        0,
                    ]
                )
                grid_dim_low = np.zeros(shape=(self.occupancy_grid_dim,))
                self.obs_space_low = np.concatenate(
                    (obs_space_low, map_dim_low, grid_dim_low)
                ).astype(np.float32)

                obs_space_hi = np.array(
                    [WINDOW_W, WINDOW_H, 1.0, 1.0, 50.0, 50.0, np.pi / 2]
                )

                obs_space_hi = np.tile(obs_space_hi, self.seq_length)
                map_dim_hi = np.array(
                    [
                        WINDOW_W,
                        WINDOW_H,
                        2 * np.pi,
                        WINDOW_W,
                        WINDOW_H,
                    ]
                )
                grid_dim_hi = np.ones(shape=(self.occupancy_grid_dim,))
                self.obs_space_hi = np.concatenate(
                    (obs_space_hi, map_dim_hi, grid_dim_hi)
                ).astype(np.float32)

        elif self.obs_type == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )
        else:
            raise NotImplementedError(
                "Unknown observation space type: %s" % self.obs_type
            )

        self.observation_space = spaces.Box(
            self.obs_space_low, self.obs_space_hi, dtype=np.float32
        )

        self.observation_dim = self.observation_space.shape[0]
        self.obs_space_range = self.obs_space_hi - self.obs_space_low

    def __init__(
        self,
        render_mode="headless",
        seq_length=1,
        obs="discrete",
        control="joystick",
        map_config="/home/eleyng/diffusion_policy/diffusion_policy/env/cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        load_map=None,
        occupancy_grid=False,
        run_mode="demo",
        run_name="mbrl",
        ep=0,
        dt=1 / 30,
        state_dim=7,
        max_num_obstacles=3,
        max_num_env_steps=1000,
        include_interaction_forces_in_rewards=False,
        set_obs=None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            render_mode:        (str) "gui" or "headless".
                                Set to "headless" for RL training without display, "gui" for data collection.
            obs:                (str) "discrete" or "rgb". Set to "discrete" for explicit state representation,
                                (str) "rgb" for image-based state representation (WARNING: not tested).
            control:            (str) "joystick" or "keyboard" for joystick or keyboard control.
            map_config:         (str) path to map configuration file in cooperative_transport/gym_table/config/maps.
                                Should be a yaml file that lists potential obstacles and their configurations.
            load_map:           (str) path to map configuration file for loading map configuration from a specific run.
            run_mode:           (str) "demo" or "eval". Set to "demo" for data collection,
                                "eval" for evaluation (loading map configuration from a specific run)
            run_name:           (str) set the custom name of the run
            ep:                 (int) episode number. Set to custom value if desired (for dataset collection).
            dt:                 (float) time step
            state_dim:          (int) dimension of state
            max_num_obstacles:  (int) maximum number of obstacles
            max_num_env_steps:  (int) maximum number of environment steps per episode
        """
        self.set_obs = set_obs
        self.ep_length = max_num_env_steps
        self.state_dim = state_dim
        self.seq_length = seq_length
        self.render_mode = render_mode
        self.occupancy_grid = occupancy_grid
        if self.occupancy_grid:
            self.grid = None
        self.dist2wall_list = None
        self.dist2wall = None
        self.avoid = None
        self.done = None
        self.delta_t = dt
        self.n = None
        self.velocity_cap = None
        self.cap = None
        self.n_step = None
        self.observation = None
        self.prev_time = None
        self.success = None
        self.done_list = None
        self.sprite_list = None
        self.target = None
        self.wallpts = None
        self.dist2obs = None
        self.dist2goal = None
        self.goal = None
        self.obs_sprite = None
        self.obstacles = None
        self.visible_obs = None
        self.num_obstacles = None
        self.max_num_obstacles = max_num_obstacles
        self.player_2 = None
        self.player_1 = None
        self.table = None
        self.inter_f = None
        self.dirname_vis_ep = None
        self.file_name_fluency = None
        self.file_name = None
        self.ep = None
        self.viewer = None
        self.screen = None
        self.base_dirname = None
        self.dirname_fluency = None
        self.dirname_vis = None
        self.control_type = None
        self.map_cfg = None
        self.obs_type = None
        self.dirname = None
        self.obs_space_dim = None
        self.observation_space = None
        self.obs_space_range = None
        self.obs_space_low = None
        self.obs_space_hi = None
        self.prediction = None
        self.ground_truth_states = None
        self.past_states = None
        self.completed_traj = None
        self.completed_traj_fluency = None
        self.include_interaction_forces_in_rewards = (
            include_interaction_forces_in_rewards
        )

        self.init_pygame()
        self.handle_kwargs(obs, control, map_config, ep)

        self.data = []
        # synthetic data
        self.init_data_paths(map_config, run_mode, run_name)
        self.interact_mode = True

        self.cumulative_reward = 0  # episode's cumulative reward
        self.fluency = {
            "inter_f": [],
            "f_del": [],
            "h_idle": [],
            "r_idle": [],
            "conf": [],
        }

        self.load_map = load_map
        self.run_mode = run_mode
        self.scale = 100

        self.init_rnd_seed_space()
        if self.run_mode == "demo":
            self.vectorized = False
        else:
            self.vectorized = True
        self.init_action_space()
        self.init_observation_space()
        self.init_env(self.load_map)

    def _set_action(self, action):
        if self.control_type == "keyboard":
            return set_action_joystick(
                action
            )  # FIXED: actions converted to continuous before being passed to set_action_joystick
        elif self.control_type in ["joystick", "data", "policy"]:
            return set_action_joystick(action)

    def start_eval(self):
        self.eval_mode = True

    def stop_eval(self):
        self.eval_mode = False

    def step(
        self, action: List[np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Union[bool, float]]]:
        """Step the environment forward.

        Parameters
        ----------
        action : List[np.ndarray], shape=(4)
            Actions of each the RL agent in UNSCALED FORM i.e. [-1., 1.]. Represents a force vector f in R^2.

        Returns
        -------
        observation : np.ndarray, shape=(self.state_dim, )
            Observations. See get_state() for details.
        reward : float
            Reward at this step.
        done : bool
            Boolean indicating whether the episode satisfies an end condition.
        info : Dict[str, Union[bool, float]]
            Information about the run. For now, contains the reward and done bool.
        """
        self.n_step += 1
        debug_print("dt in step: ", self.delta_t)
        action_for_update = self._set_action(action)

        if action is not None:
            f1, f2, x, y, angle, vx, vy, angle_speed = self.table.update(
                action_for_update,
                self.delta_t,
            )

        self.player_1.f = f1  # note: is action scaled by policy scaling factor!
        self.player_2.f = f2  # note: is action scaled by policy scaling factor!
        self.table.x = x
        self.table.y = y
        self.table.angle = angle
        self.table.x_speed = vx
        self.table.y_speed = vy
        self.table.angle_speed = angle_speed

        info = {}

        table_state = np.expand_dims(
            np.array([self.table.x, self.table.y, self.table.angle]), axis=0
        )

        self.update_metrics(table_state)
        self.observation = self.get_state()
        self.done = self.check_collision()
        self.compute_fluency(action)
        reward = self.compute_reward(
            np.expand_dims(self.observation, axis=0),
            vectorized=self.vectorized,
            collision=self.done,
            success=self.success,
            u_r=f1,
            u_h=f2,
        )
        self.cumulative_reward += reward
        debug_print(
            "Done? ",
            self.done,
            " Success? ",
            self.success,
            "Cumulative r: ",
            self.cumulative_reward,
        )

        # compute reward BEFORE dumping data to make sure terminal reward is recorded
        self.data.append(
            [self.table.x, self.table.y, self.table.angle]  # pos
            + [self.table.x_speed, self.table.y_speed, self.table.angle_speed]  # vel
            + [action]  # action
            + [reward]  # reward
            + [self.done]  # done
            + [self.success]
            + [self.n_step]  # step
            + [self.delta_t]  # dt
            + [list(self.goal)]  # goal
            + list(self.obstacles)  # obs
            + list(self.wallpts)  # wallpts
            + [self.cumulative_reward]  # cumulative reward task
        )

        info["step"] = self.n_step
        info["reward"] = reward
        info["done"] = self.done
        info["success"] = self.success

        # dump data upon episode complete
        if self.done:
            np.savez(self.file_name_fluency, **self.fluency)
            np.savez(self.config_file_name, **self.config_params)
            pickle.dump(self.data, open(self.file_name, "wb"))
            debug_print("Data saved!")
        else:
            self.redraw()

        # return self.observation, reward, self.done, info
        ### TODO: fix for co-gail
        self.linspace = np.linspace(-0.8, 0.8, 5)
        self.pivot = torch.FloatTensor(
            np.array([[i, j] for i in self.linspace for j in self.linspace])
        ).view(
            -1, 2
        )  ## TODO: figure out what this is
        self.pivot_num = len(self.pivot)
        self.pivot_id = 0

        replay = False
        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(
            np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
        ).view(1, 2)
        self.random_variable = (
            self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise
        )

        # return self.get_state(), torch.FloatTensor([[float(reward)],]), [self.done], [info], self.random_variable
        # return self.get_state(), torch.FloatTensor([[float(self.success)],]), [self.done], [info], self.random_variable

        # returns obs, reward, success, done, infos, random_seed

        if self.success:
            print(
                "Success. ",
                "Done: ",
                self.done,
                ", Success: ",
                self.success,
                ", Steps: ",
                self.n_step,
                ", Cumulative reward: ",
                self.cumulative_reward,
            )
            self.completed_traj = self.data
            self.completed_traj_fluency = self.fluency
            states = self.get_state()
            info["fluency"] = self.fluency
            if self.run_mode == "demo":
                _ = self.reset()
            return (
                states,
                reward,
                True,
                info,
            )
        elif self.done:
            print(
                "Collision. ",
                "Done: ",
                self.done,
                ", Success: ",
                self.success,
                ", Steps: ",
                self.n_step,
                ", Cumulative reward: ",
                self.cumulative_reward,
            )
            self.completed_traj = self.data
            self.completed_traj_fluency = self.fluency
            states = self.get_state()
            info["fluency"] = self.fluency
            if self.run_mode == "demo":
                _ = self.reset()
            return (
                states,
                reward,
                True,
                info,
            )
        else:
            if self.n_step > self.ep_length:
                print(
                    "Break due to step limit. ",
                    "Done: ",
                    False,
                    ", Success: ",
                    self.success,
                    ", Steps: ",
                    self.n_step,
                    ", Cumulative reward: ",
                    self.cumulative_reward,
                )
                self.completed_traj = self.data
                self.completed_traj_fluency = self.fluency
                states = self.get_state()
                info["fluency"] = self.fluency
                if self.run_mode == "demo":
                    _ = self.reset()
                return (
                    states,
                    reward,
                    True,
                    info,
                )
            else:
                info["fluency"] = self.fluency
                return (
                    self.get_state(),
                    reward,
                    False,
                    info,
                )

    def compute_fluency(self, action):
        if self.control_type == "keyboard":
            return self.compute_fluency_cont(
                action
            )  # FIXED: converted action from discrete to continuous
        elif self.control_type == "joystick":
            return self.compute_fluency_cont(action)

    def tf_w2ego(self, vec):
        """Transforms a vector from table frame to ego frame."""
        return np.array(
            [
                vec[0] * np.cos(self.table.angle) + vec[1] * np.sin(self.table.angle),
                -vec[0] * np.sin(self.table.angle) + vec[1] * np.cos(self.table.angle),
            ]
        )

    def compute_fluency_cont(self, action) -> float:
        # Interaction forces calculated from:
        # Kumar, et. al. "Force Distribution in Closed Kinematic Chains", IEEE Journal of Robotics and Automation, 1988.)
        # Interaction forces = (F_1 - F_2) dot (r_1 - r_2)

        self.inter_f = (self.player_1.f - self.player_2.f) @ (
            self.table.table_center_to_player1 - self.table.table_center_to_player2
        )
        self.fluency["inter_f"].append(self.inter_f)
        # Human Idle: if all actions are 0, then it is idle
        if not np.any(self.player_2.f):
            self.fluency["h_idle"].append(1)
        else:
            self.fluency["h_idle"].append(0)
        # Robot Idle
        if not np.any(self.player_1.f):
            self.fluency["r_idle"].append(1)
        else:
            self.fluency["r_idle"].append(0)
        # Concurrent action: when both are acting
        if np.any(self.player_2.f) and np.any(self.player_1.f):
            self.fluency["conf"].append(1)
        else:
            self.fluency["conf"].append(0)
        # Funct. delay: when both are not acting
        if (not np.any(self.player_2.f)) and (not np.any(self.player_1.f)):
            self.fluency["f_del"].append(1)
        else:
            self.fluency["f_del"].append(0)

    def compute_fluency_disc(self, action):
        if action >= 9:
            self.fluency["conf"].append(1)
        else:
            self.fluency["conf"].append(0)
        if action == 0:
            self.fluency["f_del"].append(1)
        else:
            self.fluency["f_del"].append(0)
        if action in [1, 2, 3, 4]:
            self.fluency["r_idle"].append(1)
        else:
            self.fluency["r_idle"].append(0)
        if action in [5, 6, 7, 8]:
            self.fluency["h_idle"].append(1)
        else:
            self.fluency["h_idle"].append(0)
        if action in [10, 13, 20, 23]:
            self.fluency["inter_f"].append(1)
        else:
            self.fluency["inter_f"].append(0)

    def update_metrics(self, states):
        # dist2goal
        self.dist2goal = np.linalg.norm(states[:, :2] - self.goal, axis=1)
        self.dist2obs = np.asarray(
            [
                np.linalg.norm(
                    states[:, :2]
                    - np.array([self.obs_sprite[i].x, self.obs_sprite[i].y]),
                    axis=1,
                )
                for i in range(self.num_obstacles)
            ],
            dtype=np.float32,
        )

        self.avoid = np.array(self.obstacles) - np.array([self.table.x, self.table.y])

        # dist2wall
        self.dist2wall_list = np.vstack(
            (
                states[:, 0],
                WINDOW_W - states[:, 0],
                states[:, 1],
                WINDOW_H - states[:, 1],
            )
        ).T

        self.dist2wall = np.min(self.dist2wall_list, axis=1, keepdims=True)

    def compute_reward(
        self,
        states,
        interaction_forces=None,
        vectorized=False,
        collision=None,
        success=None,
        u_r=None,
        u_h=None,
    ) -> float:
        reward = custom_reward_function(
            states,
            self.goal,
            self.obstacles,
            interaction_forces=self.include_interaction_forces_in_rewards,
            vectorized=vectorized,
            collision=collision,
            success=success,
            u_r=u_r,
            u_h=u_h,
        )
        return reward

    def check_collision(self) -> bool:
        """Check for collisions.

        Returns
        -------
        collided : Boolean
            Whether the table has collided with the obstacles
        """
        hit_list = pygame.sprite.spritecollide(
            self.table, self.done_list, False, pygame.sprite.collide_mask
        )

        if any(hit_list):
            if any(
                pygame.sprite.spritecollide(
                    self.table, [self.target], False, pygame.sprite.collide_mask
                )
            ):
                self.success = True
                debug_print("HIT TARGET")
            else:
                debug_print("HIT OBSTACLE")
            return True  # , reward
        else:
            # wall collision
            if not self.screen.get_rect().contains(self.table):
                debug_print("HIT WALL")
                return True
            else:
                return False  # , 0

    def reset(self, load_map=None) -> np.ndarray:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray, shape=(state_dim,)
            Observation. TODO: make this not return actions and dist2stuff.
        """

        debug_print("Reset episode.\n")

        self.cumulative_reward = 0  # episode's cumulative reward
        self.fluency = {
            "inter_f": [],
            "f_del": [],
            "h_idle": [],
            "r_idle": [],
            "conf": [],
        }

        self.init_env(load_map)

        self.ep += 1
        self.file_name = os.path.join(self.dirname, "ep_" + str(self.ep) + ".pkl")
        self.config_file_name = os.path.join(
            self.map_config_dir, "ep_" + str(self.ep) + ".npz"
        )
        self.file_name_fluency = os.path.join(
            self.dirname_fluency, "ep_" + str(self.ep) + ".npz"
        )
        if not os.path.exists(os.path.dirname(self.file_name_fluency)):
            os.makedirs(os.path.dirname(self.file_name_fluency))
        if not os.path.exists(os.path.dirname(self.file_name)):
            os.makedirs(os.path.dirname(self.file_name))
        if not os.path.exists(os.path.dirname(self.config_file_name)):
            os.makedirs(os.path.dirname(self.config_file_name))

        self.dirname_vis_ep = os.path.join(
            self.dirname_vis, "ep_" + str(self.ep) + "_images"
        )
        if not exists(self.dirname_vis_ep):
            debug_print("Making image directory: ", self.dirname_vis)
            mkdir(self.dirname_vis_ep)

        self.data = []

        ### TODO: fix for co-gail
        self.linspace = np.linspace(-0.8, 0.8, 5)
        self.pivot = torch.FloatTensor(
            np.array([[i, j] for i in self.linspace for j in self.linspace])
        ).view(
            -1, 2
        )  ## TODO: figure out what this is
        self.pivot_num = len(self.pivot)
        self.pivot_id = 0

        replay = False
        mbrl = True
        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(
            np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
        ).view(1, 2)
        self.random_variable = (
            self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise
        )
        # print("code:", self.random_variable)

        # if replay:
        #     human_action_return = torch.FloatTensor([[0.0, 0.0]])
        #     return self.get_state(), self.random_variable, human_action_return
        # else:
        #     return self.get_state(), self.random_variable
        return self.get_state()  # , self.done

    def mp_check_collision_and_success(self, state) -> bool:
        """Check for collisions and success.

        Returns
        -------
        collided : Boolean
            Whether the table has collided with the obstacles
        success : Boolean
            Whether the table has reached the target
        """
        # set table position
        self.table.x = state[0]
        self.table.y = state[1]
        self.table.angle = state[2]
        # update sprite
        self.table.image = pygame.transform.rotate(
            self.table.original_img, np.degrees(self.table.angle)
        )
        self.table.rect = self.table.image.get_rect(center=(self.table.x, self.table.y))
        self.table.mask = pygame.mask.from_surface(self.table.image)

        hit_list = pygame.sprite.spritecollide(
            self.table, self.done_list, False, pygame.sprite.collide_mask
        )

        collision = False
        success = False

        if any(hit_list):

            collision = True
            if any(
                pygame.sprite.spritecollide(
                    self.table, [self.target], False, pygame.sprite.collide_mask
                )
            ):
                success = True
                debug_print("HIT TARGET")
            else:
                debug_print("HIT OBSTACLE")
        else:
            # wall collision
            if not self.screen.get_rect().contains(self.table):

                collision = True
                debug_print("HIT WALL")

        return collision, success

    @staticmethod
    def standardize(self, ins, mean, std):
        s = np.divide(np.subtract(ins, mean), std)
        return s

    @staticmethod
    def unstandardize(self, ins, mean, std):
        us = np.multiply(ins, std).add(mean)
        return us

    def update_prediction(self, pred):
        self.prediction = pred

    def draw_gt(self, gt):
        self.ground_truth_states = gt

    def draw_past_states(self, past_states):
        self.past_states = past_states

    def redraw(self) -> None:
        """Updates the pygame visualization."""
        self.screen.fill((BLACK))
        # Update table image
        self.sprite_list.draw(self.screen)
        if self.prediction is not None:
            pygame.draw.circle(
                self.screen, (243, 162, 97, 1), [self.table.x, self.table.y], 5
            )

            for p in range(len(self.prediction)):
                pygame.draw.circle(
                    self.screen, (243, 162, 97, 1), self.prediction[p][:2], 3
                )
        if self.ground_truth_states is not None:
            for p in range(len(self.ground_truth_states)):
                pygame.draw.circle(
                    self.screen, (42, 157, 142, 0.5), self.ground_truth_states[p][:2], 2
                )
        if self.past_states is not None:
            for p in range(len(self.past_states)):
                pygame.draw.circle(
                    self.screen, (42, 157, 142, 0.5), self.past_states[p][:2], 2
                )

        pygame.display.update()

    def render(self, mode: str = "human") -> Union[np.ndarray, None]:
        """Renders an image.

        Parameters
        ----------
        mode : str, default="human"
            Render modes. Can be "human" or "rgb_array".

        Returns
        -------
        output : Union[np.ndarray, None]
            If mode is "human", then return nothing and update the viewer.
            If mode is "rgb_array", then return an image as np array to be rendered.
        """
        img4disp = self.get_image()

        if self.n_step % 10 == 0:
            img = cv.cvtColor(img4disp, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((WINDOW_W, WINDOW_H))
            # img.save(os.path.join(self.dirname_vis_ep, str(self.n_step) + ".png"))

        if mode == "human":
            pass
            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            #     self.viewer.imshow(img)
        elif mode == "rgb_array":
            return img4disp
        else:
            raise NotImplementedError

    @staticmethod
    def get_image() -> np.ndarray:
        """Gets the pygame display img for render.

        Returns
        -------
        img : np.ndarray, shape=(H, W, 3)
            3-channel image of the environment.
        """
        # observation
        img = np.fliplr(
            np.flip(
                np.rot90(
                    pygame.surfarray.array3d(pygame.display.get_surface()).astype(
                        np.uint8
                    )
                )
            )
        )
        return img

    def _get_rgb_state(self):
        """Gets the player state.

        Returns
        -------
        state : np.ndarray, shape=(STATE_H, STATE_W, 3)
            The state created after resizing the rendered image from pygame
        """
        img = self.get_image()
        img = Image.fromarray(img)
        img = img.resize((STATE_H, STATE_W))
        return np.asarray(img)

    def get_state(self) -> np.ndarray:
        if self.obs_type == "discrete":
            if not self.occupancy_grid:
                return self._get_discrete_state()
            else:
                return self._get_discrete_state_map()
        elif self.obs_type == "rgb":
            return self._get_rgb_state()

    def _get_discrete_state(self) -> np.ndarray:
        """Gets the player state.

        Returns
        -------
        state : np.ndarray, shape=(state_dim, )
            The state.
        """
        state = np.zeros(shape=(self.state_dim,), dtype=np.float32)
        state[0] = self.table.x
        state[1] = self.table.y
        state[2] = np.cos(self.table.angle)  # self.table.angle  #
        state[3] = np.sin(self.table.angle)  # self.target.x  #
        state[4] = self.table.x_speed
        state[5] = self.table.y_speed
        state[6] = self.table.angle_speed

        self.obs_hist[: -self.state_dim] = self.obs_hist[self.state_dim :]
        self.obs_hist[-self.state_dim :] = state

        self.full_observation = np.concatenate(
            (self.obs_hist, self.map_info, self.grid)
        )

        return torch.from_numpy(self.full_observation).float()

    def _get_discrete_state_map(self) -> np.ndarray:
        """Gets the player state.

        Returns
        -------
        state : np.ndarray, shape=(self.state_dim + occ_grid_dim)
            The state.
        """
        state = np.zeros(
            shape=(self.state_dim,), dtype=np.float32
        )  # self.occupancy_grid is the dimension of the self.WINDOW_W * self.WINDOW_H / (self.occ_grid_scale ** 2), where self.occ_grid_scale = 10
        state[0] = self.table.x
        state[1] = self.table.y
        state[2] = np.cos(self.table.angle)  # self.table.angle  #
        state[3] = np.sin(self.table.angle)  # self.target.x  #
        state[4] = self.table.x_speed
        state[5] = self.table.y_speed
        state[6] = self.table.angle_speed

        self.obs_hist[: -self.state_dim] = self.obs_hist[self.state_dim :]
        self.obs_hist[-self.state_dim :] = state

        self.full_observation = np.concatenate(
            (self.obs_hist, self.map_info, self.grid)
        )

        return torch.from_numpy(self.full_observation).float()

    def make_occupancy_grid(
        self, WINDOW_W=1200, WINDOW_H=600, buffer_size=50, obs_size=100, scale=100
    ):
        ### UNCOMMENT FOR OCCUPANCY GRID
        obs_w = obs_size / WINDOW_W
        obs_h = obs_size / WINDOW_H
        buffer_w = buffer_size / WINDOW_W
        buffer_h = buffer_size / WINDOW_H
        map_dim_w = int(WINDOW_W / scale)
        map_dim_h = int(WINDOW_H / scale)
        occGrid = np.zeros(map_dim_w * map_dim_h)
        gridPointsRange_w = np.linspace(0, 1, num=map_dim_w)
        gridPointsRange_h = np.linspace(0, 1, num=map_dim_h)
        # process obstacle data into occupancy grid
        occGridSamples = np.zeros([map_dim_w * map_dim_h, 2])
        idx = 0
        for i in gridPointsRange_w:
            for j in gridPointsRange_h:
                occGridSamples[idx, 0] = i
                occGridSamples[idx, 1] = j
                # print(occGridSamples[idx,:])
                idx += 1
        scaled_obs = []
        for idx in range(self.obstacles.flatten().shape[0]):
            if idx % 2 == 0:
                osc = self.obstacles.flatten()[idx] / WINDOW_W
            else:
                osc = self.obstacles.flatten()[idx] / WINDOW_H
            scaled_obs.append(osc)
        scaled_obs_lst = np.asarray(scaled_obs, dtype=np.float32)
        occGrid = np.zeros(map_dim_w * map_dim_h)
        for i in range(0, map_dim_w * map_dim_h):
            # print("checking:", i, occGridSamples[i, :])
            occGrid[i] = self.isSampleFree(
                occGridSamples[i, :],
                scaled_obs_lst,
                obs_dim=2,
                obs_w=obs_w,
                obs_h=obs_h,
                buffer_w=buffer_w,
                buffer_h=buffer_h,
            )
        occGrid = np.asarray(occGrid, dtype=np.float32)
        return occGrid

    def isSampleFree(
        self,
        grid_position,
        obs_pos_lst,
        obs_dim=2,
        obs_w=None,
        obs_h=None,
        buffer_w=None,
        buffer_h=None,
    ):
        # loop thru each obstacle, return 0 for the grd pos if occupied (incl buffer consideration)
        for o in range(0, int(obs_pos_lst.shape[0] / obs_dim)):
            # check obs boundaries -- x
            x_lo = grid_position[0] >= (obs_pos_lst[o * obs_dim] - obs_w - buffer_w)
            x_hi = grid_position[0] <= (obs_pos_lst[o * obs_dim] + obs_w + buffer_w)
            y_lo = grid_position[1] >= (obs_pos_lst[o * obs_dim + 1] - obs_h - buffer_h)
            y_hi = grid_position[1] <= (obs_pos_lst[o * obs_dim + 1] + obs_h + buffer_h)

            if x_lo and x_hi and y_lo and y_hi:
                return 0
        return 1
