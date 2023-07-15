import gym
import pygame
import time
import matplotlib
import argparse
from gym import logger
import numpy as np

import cooperative_transport.gym_table.envs.utils as utils

FPS = 30
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS


try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn(f"failed to set matplotlib backend, plotting will not work: {str(e)}")
    plt = None

from collections import deque
from pygame.locals import VIDEORESIZE

VERBOSE = True


def debug_print(*args):
    if not VERBOSE:
        return
    print(*args)


def init_joystick():
    pygame.joystick.init()
    # Get count of joysticks.
    joystick_count = pygame.joystick.get_count()
    debug_print("Number of joysticks: {}".format(joystick_count))
    joysticks = []
    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        joysticks.append(joystick)

        try:
            jid = joystick.get_instance_id()
        except AttributeError:
            # get_instance_id() is an SDL2 method
            jid = joystick.get_id()
        debug_print("Joystick {}".format(jid))

        # Get the name from the OS for the controller/joystick.
        name = joystick.get_name()
        debug_print("Joystick name: {}".format(name))

        try:
            guid = joystick.get_guid()
        except AttributeError:
            # get_guid() is an SDL2 method
            pass
        else:
            debug_print("GUID: {}".format(guid))
    return joysticks


def play(env, transpose=True, fps=60, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    obs = env.reset()
    rendered = env.render(mode="rgb_array")

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            keys_to_action = utils.get_keys_to_action()
        # else:
        #     assert False, (
        #         env.spec.id
        #         + " does not have explicit key to action mapping, "
        #         + "please specify one manually"
        #     )
    debug_print("keys to action", keys_to_action)
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    # env_done = True
    env_done = False

    screen = pygame.display.set_mode(video_size)
    next_game_tick = time.time()
    clock = pygame.time.Clock()
    cnt = 0
    joysticks = init_joystick()
    # GAME LOOP
    while running:
        loops = 0
        cnt += 1
        if env_done:
            env_done = False
            obs = env.reset()
            time.sleep(1)

            debug_print("env_don")
            # break
        else:
            while time.time() > next_game_tick and loops < MAX_FRAMESKIP:
                # UPDATE GAME
                if env.control_type == "keyboard":
                    action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                elif env.control_type == "joystick":
                    p1_id = 0  # player 1 is blue
                    # FIXME: for now only have 1 joystick, so using same for both
                    p2_id = 1

                    action = np.array(
                        [
                            joysticks[p1_id].get_axis(0),
                            joysticks[p1_id].get_axis(1),
                            joysticks[p2_id].get_axis(0),
                            joysticks[p2_id].get_axis(1),
                        ]
                    )
                debug_print("ACTION: ", action)

                obs, rew, env_done, info = env.step(action)
                debug_print("loop: ", loops, info, "\n\, obs: ", obs)
                next_game_tick += CONST_DT
                loops += 1
                if env_done:
                    break

            ##if callback is not None:
            ##    callback(prev_obs, obs, action, rew, env_done, info)
            # CLOCK TICK
            clock.tick(FPS)
            if clock.get_fps() > 0:
                debug_print("Reported dt: ", 1 / clock.get_fps())

        if obs is not None:
            rendered = env.render(mode="rgb_array")
            # display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    debug_print("REGISTERED KEY PRESS")
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                debug_print(video_size)

        # pygame.display.flip()

        # clock.tick(fps)

    pygame.quit()


class PlayPlot:
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(
                range(xmin, xmax), list(self.data[i]), c="blue"
            )
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="cooperative_transport.gym_table:table-v0",
        help="Define Environment",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="discrete",
        help="Define Observation Space, discrete/rgb",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="joystick",
        help="Define Control Input, keyboard/joystick",
    )
    parser.add_argument(
        "--map_config",
        type=str,
        default="cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        help="Map Config File Path",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="success_1",
        help="strat",
    )

    parser.add_argument(
        "--run_mode",
        type=str,
        default="demo",
        help="run mode",
    )

    parser.add_argument(
        "--ep",
        type=int,
        default=207,
        help="ep",
    )
    args = parser.parse_args()
    env = gym.make(
        args.env,
        obs=args.obs,
        control=args.control,
        map_config=args.map_config,
        run_mode=args.run_mode,
        strategy_name=args.strategy,
        ep=args.ep,
        dt=CONST_DT,
        physics_control_type="force",
    )

    play(env, zoom=1, fps=30)


if __name__ == "__main__":
    main()
