import sys
from os import mkdir, listdir
from os.path import join, exists, dirname, abspath, isdir
import math
import time

# set OMPL install path or add path to system path directly
sys.path.insert(0, join('/home/collab1/table-carrying-sim/ompl-1.5.2', 'py-bindings'))
sys.path.insert(0, join('/home/collab1/table-carrying-sim/third_party', 'RobotMP'))
sys.path.insert(0, join('/home/collab1/table-carrying-sim'))

import gym
import numpy as np
from numpy.linalg import norm
import cv2
import pygame
# from simple_pid import PID

from cooperative_transport.gym_table.envs.utils import load_cfg
import third_party.RobotMP.robotmp as rmp


FPS = 30
CONST_DT = 1 / FPS

env = None

RENDER_LIVE = False

# get metadata from config file
yaml_filepath = join("cooperative_transport/gym_table/config/inference_params.yml")
meta_cfg = load_cfg(yaml_filepath)
dataset = meta_cfg["dataset"]
trained_model = meta_cfg["trained_model"]
save_dir = meta_cfg["save_dir"]
save_dir = save_dir + "-" + dataset
map_cfg = meta_cfg["map_config"]  # path to map config file

# if loading trajectories from file, then load them here
root = join("dataset", dataset)

FILES = [
    join(root, sd, ssd)
    for sd in listdir(root)
    if isdir(join(root, sd))
    for ssd in listdir(join(root, sd))
]


def is_safe(state):
    """
    customize state validity checker function
    Input: current state
    Output: True if current state is valid (collision free, etc.)
    """
    return not env.mp_check_collision(state)


def make_env():
    global env
    env = gym.make(
        "cooperative_transport.gym_table:table-v0",
        obs="discrete",
        control="keyboard",
        map_config="cooperative_transport/gym_table/config/maps/one_obstacle.yml",
        run_mode="demo",
        strategy_name="rrt",
        ep=0,
        dt=CONST_DT,
        physics_control_type="velocity",
    )
    # env.configure_data(data_playback=False)


def mppi(
    control_seq, trajectory, max_iter, n=40, n_samples=10, noise_range=0.3, alpha=1.0
):
    def sample_noise(num=n):
        return np.random.uniform(low=-noise_range, high=noise_range, size=(num, 4))

    def compute_reward(input_seq):
        # TODO execute input_seq in sim, get trajectory L2 reward
        # reset sim state
        env.reset_table_state()

        executed_trajectory = np.zeros((n, 3))
        # apply action
        for step in range(n):
            action = input_seq[step, :].reshape(2, 2)
            f1, f2, x, y, angle, vx, vy, angle_speed = env.table.update(
                action, env.delta_t, -1, False, update_image=False
            )
            executed_trajectory[step, :] = [x, y, angle]
        # compute L2 norm
        dist = np.mean(np.linalg.norm(trajectory - executed_trajectory, axis=1))
        return -dist

    best_seq = control_seq.copy()
    best_score = compute_reward(best_seq)

    for it in range(max_iter):
        curr_seq = control_seq.copy()
        for t in range(n):
            noises = []
            scores = []
            weighted_sum = 0.0
            # rollout
            for s in range(n_samples):
                noise = sample_noise(n - t)
                new_seq = curr_seq.copy()
                new_seq[t:, :] += noise
                new_seq = np.clip(new_seq, -1.0, 1.0)
                score = compute_reward(new_seq)
                noises.append(noise)
                scores.append(score)
                weighted_sum += np.exp((1.0 / alpha) * score)
            for s in range(n_samples):
                curr_seq[t, :] += (
                    np.exp((1.0 / alpha) * scores[s]) * noises[s][0, :] / weighted_sum
                )
        # evaluate
        prev_score = compute_reward(control_seq)
        new_score = compute_reward(curr_seq)
        print(f"prev_score: {prev_score} new_score: {new_score}")
        if new_score > prev_score:
            control_seq = curr_seq
            best_seq = curr_seq
            best_score = new_score
            print("New best score: ", best_score)
        if it % 10 == 0:
            print("Iteration: ", it)
    return best_seq


def pid(
    trajectory,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    max_iter=40,
    dt=CONST_DT,
    eps=1e-2,
    linear_speed_limit=[-2.0, 2.0],
    angular_speed_limit=[-np.pi / 4, np.pi / 4],
):
    def get_action_from_wrench(wrench):
        G = (
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [
                -np.sin(wrench[2]),
                -np.cos(wrench[2]),
                np.sin(wrench[2]),
                np.cos(wrench[2]),
            ],
        )
        F_des = np.linalg.pinv(G).dot(wrench)
        # F_des = np.clip(F_des, -1.0, 1.0)
        f1_x, f1_y, f2_x, f2_y = F_des[0], F_des[1], F_des[2], F_des[3]

        return f1_x, f1_y, f2_x, f2_y

    env.reset_table_state()

    num_waypoints = trajectory.shape[0]
    curr_state = np.array([env.table.x, env.table.y, env.table.angle])
    last_state = None
    last_control = None
    last_error = None

    for i in range(num_waypoints):
        curr_target = trajectory[i]
        step = 0
        curr_state = np.array([env.table.x, env.table.y, env.table.angle])
        last_state = None
        last_control = None
        last_error = None
        proportional = 0
        integral = 0
        derivative = 0
        while np.linalg.norm(curr_state - curr_target) > eps and step < max_iter:
            step += 1
            # print("Target: ", curr_target, " State: ", curr_state)
            error = curr_target - curr_state
            integral += error * dt
            d_error = 0 if last_error is None else (error - last_error) / dt
            wrench = kp * error + ki * integral + kd * d_error
            last_error = error
            f1_x, f1_y, f2_x, f2_y = get_action_from_wrench(wrench)

            # curr_state = env.table.update_with_wrench(action, env.delta_t)
            # env.table.image = pygame.transform.rotate(
            #     env.table.original_img, env.table.angle
            # )
            # env.table.rect = env.table.image.get_rect(center=(env.table.x, env.table.y))
            # env.table.mask = pygame.mask.from_surface(env.table.image)
            # env.redraw()
            # time.sleep(0.025)

            # d_state = curr_state - (last_state if last_state is not None else curr_state)
            # d_error = error - (last_error if last_error is not None else error)
            # proportional -= kp * error
            # integral += ki * error * dt
            # integral = clamp_action(integral)
            # derivative = -kd * d_state / dt
            # action = proportional + integral + derivative
            # action = clamp_action(action)
            # last_state = curr_state
            # last_control = action
            # last_error = error
            # curr_state = env.table.update_with_wrench(
            #     action, env.delta_t)

        print("target: ", curr_target, " state: ", curr_state)


def main():

    make_env()

    lower_limits = [50, 50, -np.pi]
    upper_limits = [1150, 550, np.pi]

    # create planner
    planner = rmp.OMPLPlanner(
        state_space_bounds=[lower_limits, upper_limits],
        state_validity_checker=is_safe,
        planner="rrt_star",
    )

    start_state = [env.table.x, env.table.y, env.table.angle]
    goal_state = [env.goal[0], env.goal[1], np.pi / 2]
    # set start and goal states
    print("start: ", start_state)
    print("goal: ", goal_state)

    planner.set_start_and_goal(start=start_state, goal=goal_state)
    planner.set_step_distance(5)

    # solve!
    path, cost, t = planner.plan(time_limit=1.0)
    trajectory = None
    # exit(0)
    if RENDER_LIVE:
        rendered = env.render(mode="rgb_array")
        video_size = [rendered.shape[1], rendered.shape[0]]
        screen = pygame.display.set_mode(video_size)
        if path is not None:
            print(dir(path))
            print(path.getStateCount())
            for idx in range(path.getStateCount()):
                print(
                    path.getState(idx)[0], path.getState(idx)[1], path.getState(idx)[2]
                )
                # update state and render
                env.table.x = path.getState(idx)[0]
                env.table.y = path.getState(idx)[1]
                env.table.angle = path.getState(idx)[2]
                angle = math.degrees(env.table.angle)
                # get a rotated image
                env.table.image = pygame.transform.rotate(env.table.original_img, angle)
                env.table.rect = env.table.image.get_rect(
                    center=(env.table.x, env.table.y)
                )
                env.table.mask = pygame.mask.from_surface(env.table.image)
                env.redraw()
                time.sleep(0.01)
        exit(0)
    if path is not None:
        print(f"Way points: {path.getStateCount()}")
        trajectory = np.zeros((path.getStateCount(), 3))
        for idx in range(path.getStateCount()):
            # print(path.getState(idx)[0], path.getState(idx)[1], path.getState(idx)[2])
            trajectory[idx, :] = [
                path.getState(idx)[0],
                path.getState(idx)[1],
                path.getState(idx)[2],
            ]

    # control_steps = path.getStateCount()

    # initial_control_seq = np.random.normal(0.0, 1.0, size=(control_steps, 4))
    # initial_control_seq = np.clip(initial_control_seq, -1.0, 1.0)

    # load trajectory from file
    for f in FILES:
        trajectory = dict(np.load(f))
        print(f"Loaded {f}")
        waypoints = trajectory["states"][:, :3]
        print(f"Waypoints: {waypoints.shape, waypoints[0]}")

        # PID solver to follow trajectory from RRT
        # pid(
        #     waypoints,
        #     kp=1.0,
        #     ki=1.0,
        #     kd=0.5,
        #     max_iter=3,
        #     eps=2.0,
        #     linear_speed_limit=[-5, 5],
        # )

    # best_control_seq = mppi(initial_control_seq, trajectory, max_iter=50, n=control_steps, n_samples=10, noise_range=0.8, alpha=1.0)
    # print(best_control_seq)

    # visualize control sequence
    # env.reset_table_state()

    # apply action
    # for step in range(control_steps):
    #     action = best_control_seq[step, :].reshape(2, 2)
    #     f1, f2, x, y, angle, vx, vy, angle_speed = env.table.update(
    #         action, env.delta_t, -1, False, update_image=False
    #     )
    #     # update state and render
    #     angle = math.degrees(env.table.angle)
    #     # get a rotated image
    #     env.table.image = pygame.transform.rotate(env.table.original_img, angle)
    #     env.table.rect = env.table.image.get_rect(center=(env.table.x, env.table.y))
    #     env.table.mask = pygame.mask.from_surface(env.table.image)
    #     env.redraw()
    #     time.sleep(0.01)


if __name__ == "__main__":
    main()
