import math
import os
import pickle
from typing import Tuple, List

import numpy as np
import pygame
from PIL import Image

# TODO: either move all this stuff into a util file as enum or expose it to be configurable
# table parameters
m = 2.0
b = 2.0
I = 1.0
L = 1.0
d = 40

WINDOW_W = 1200
WINDOW_H = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHTBLUE = (138, 181, 255)
BLUE = (20, 104, 250)
LIGHTORANGE = (255, 225, 173)
ORANGE = (255, 167, 15)

VERBOSE = False


def debug_print(*args):
    if not VERBOSE:
        return
    print(*args)


# TODO: delete this later
"""
padsize = 25
border = 5
left2 = pygame.Rect(worldx - border - padsize * 3, border + padsize, padsize, padsize)
down2 = pygame.Rect(worldx - border - padsize * 2, border + padsize * 2, padsize, padsize)
up2 = pygame.Rect(worldx - padsize * 2 - border, border, padsize, padsize)
right2 = pygame.Rect(worldx - padsize - border, padsize + border, padsize, padsize)

left1 = pygame.Rect(worldx - padsize * 6 - 3 * border, border + padsize, padsize, padsize)
down1 = pygame.Rect(worldx - padsize * 5 - 3 * border, border + padsize * 2, padsize, padsize)
up1 = pygame.Rect(worldx - padsize * 5 - 3 * border, border, padsize, padsize)
right1 = pygame.Rect(worldx - padsize * 4 - 3 * border, padsize + border, padsize, padsize)
"""


class Obstacle(pygame.sprite.Sprite):
    """Obstacle object."""

    def __init__(self, position: np.ndarray, size=(50, 50)) -> None:
        """Initialize obstacle.

        Parameters
        ----------
        position : np.ndarray, shape=(2)
            Planar position of the obstacle.
        """
        # create sprite
        pygame.sprite.Sprite.__init__(self)

        # relative paths
        dirname = os.path.dirname(__file__)
        obstacle_path = os.path.join(dirname, "images/obstacle.png")

        # visuals
        self.original_img = pygame.image.load(obstacle_path).convert()
        self.original_img = pygame.transform.scale(self.original_img, size)
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2


class Target(pygame.sprite.Sprite):
    """Target object."""

    def __init__(self, position: np.ndarray) -> None:
        """Initialize the target.

        Parameters
        ----------
        position : np.ndarray, shape=(2)
            Planar position of the target.
        """
        # relative paths
        dirname = os.path.dirname(__file__)
        target_path = os.path.join(dirname, "images/target.png")

        # initial conditions
        self.x = position[0]
        self.y = position[1]

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.original_img = pygame.image.load(target_path).convert()
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.rect = self.image.get_rect()

        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2
        # pygame.sprite.Sprite.__init__(self)
        # original_img = Image.open(target_path)
        # mode = original_img.mode
        # sz = original_img.size
        # data = original_img.tobytes()
        # self.original_img = pygame.image.fromstring(data, sz, mode)
        # self.image = self.original_img
        # self.rect = self.image.get_rect(center = (self.x, self.y))
        # self.mask = pygame.mask.from_surface(self.image)


class Table(pygame.sprite.Sprite):
    """Table object."""

    def __init__(
        self,
        x=0.25 * WINDOW_W,
        y=0.25 * WINDOW_H,
        angle=0.0,
        physics_control_type="force",
        length=1.0,
    ) -> None:
        """Initialize the table.

        Parameters
        ----------
        count_trajectory : int
            Number of trajectories.
        num_obstacles : int
            Number of obstacles.

        TODO: make obstacles an attribute of the environment rather than the table.
        TODO: same for trajectories.
        """
        """# TODO: replace goal creat by passing from gym
        # create goal
        # goal_angle = np.random.uniform(0, math.pi//2)
        goal_angle = math.pi / 4
        # goal_radius = np.random.uniform(300,360)
        goal_radius = 100
        # goal location
        self.goal = np.array([0.5 * WINDOW_W, 0.5 * WINDOW_H])
        self.goal += goal_radius * np.array(
            [math.cos(goal_angle), math.sin(goal_angle)]
        )
        self.dist2goal = None

        # TODO: replace obstacle create by passing from gym
        # create obstacle
        self.obs = np.zeros((num_obstacles, 2))
        for i in range(num_obstacles):
            # obs_angle = goal_angle + np.random.uniform(-math.pi / 6, math.pi / 6)
            obs_radius = 50
            obs = np.array([0.5 * WINDOW_W, 0.5 * WINDOW_H])
            # obs += obs_radius * np.array([math.cos(goal_angle), math.sin(goal_angle)])
            obs += np.array([i * obs_radius, -1 * i * obs_radius])
            self.obs[i] = obs

        """
        # debug_print("self.obs", type(self.obs), self.obs.shape, self.obs)  # DEBUG

        # defining relative paths
        dirname = os.path.dirname(__file__)
        table_img_path = os.path.join(dirname, "images/table.png")
        data_path = os.path.join(dirname, "datasets/2020-11-16_dataset.pkl")

        self.physics_control_type = physics_control_type
        # initial conditions
        self.x_speed = 0.0
        self.y_speed = 0.0
        self.angle_speed = 0.0

        self.x = x
        self.y = y
        self.angle = angle
        self.px = 0.25 * WINDOW_W
        self.py = 0.25 * WINDOW_H
        self.pangle = 0.0
        # create sprite
        pygame.sprite.Sprite.__init__(self)
        original_img = Image.open(table_img_path)
        mode = original_img.mode
        sz = original_img.size
        data = original_img.tobytes()
        self.original_img = pygame.image.fromstring(data, sz, mode)
        self.w, self.h = self.original_img.get_size()
        self.length_from_center_to_person = length / 2
        self.table_center_to_player1 = np.array([self.x + self.length_from_center_to_person * np.cos(self.angle), 
                        self.y + self.length_from_center_to_person * np.sin(self.angle)])
        self.table_center_to_player2 = np.array([self.x - self.length_from_center_to_person * np.cos(self.angle),
                        self.y - self.length_from_center_to_person * np.sin(self.angle)])
        # self.image = self.original_img
        # get a rotated image
        self.image = pygame.transform.rotate(self.original_img, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)

        # robot noise properties
        # NOTE - to turn off the noise, set sigma = 0
        self.mu = 0.0
        self.sigma = 10.0
        self.time2hold = 10
        self.noise = None

        # load trajectories (synth data)
        # data = pickle.load(open(data_path, "rb"))
        # self.data = data
        # self.count_trajectory = count_trajectory  # which trajectory in the dataset
        # self.curr_traj = data[self.count_trajectory]  # initialize trajectory
        # while len(self.curr_traj) < 200 and self.count_trajectory < len(data):
        #    # debug_print("TRUE", self.count_trajectory)  # DEBUG
        #    self.count_trajectory += 1
        #    self.curr_traj = data[self.count_trajectory]

        self.cap = 1.0
        self.cap_alpha_min = -np.pi / 8
        self.cap_alpha_max = np.pi / 8
        self.min_velocity_angle = -np.pi / 8
        self.max_velocity_angle = np.pi / 8

        self.policy_scaling_factor = 50

    def acceleration(
        self, f1: np.ndarray, f2: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute the acceleration given forces.

        Parameters
        ----------
        f1 : np.ndarray, shape=(2)
            Force 1.
        f2 : np.ndarray, shape=(2)
            Force 2.

        Returns
        -------
        a_x : float
            Linear x acceleration.
        a_y : float
            Linear y acceleration.
        a_angle : float
            Angular acceleration.
        """
        # equations of motion for table
        # debug_print('f1: ', f1.shape, f1, 'f2: ', f2.shape, f2)
        # a_x = 1.0 / m * (f1[0] + f2[0])
        a_x = -b / m * self.x_speed + 1.0 / m * (f1[0] + f2[0])
        # a_y = 1.0 / m * (f1[1] + f2[1])
        a_y = -b / m * self.y_speed + 1.0 / m * (f1[1] + f2[1])
        M_z = (
            L
            / 2.0
            * (
                math.sin(self.angle) * (f2[0] - f1[0])
                + math.cos(self.angle) * (-f1[1] + f2[1])
            )
        )  # player 1 on the right (blue tri keys), player2 on left (orange circ wasd)
        debug_print("Mz", M_z, self.angle_speed, self.angle)
        a_angle = -d / I * self.angle_speed + 1.0 / I * M_z
        debug_print("a_angle PRE clip", a_angle)
        a_angle = np.clip(a_angle, self.cap_alpha_min, self.cap_alpha_max)
        debug_print("a_angle_post clip", a_angle)
        # a_angle = 1.0 / I * M_z
        return a_x, a_y, a_angle

    def control_to_velocity(self, f1, f2):
        vx = f1[0] + f2[0]
        vy = f1[1] + f2[1]
        va = (
            L
            / 2.0
            * (
                math.sin(self.angle) * (f2[0] - f1[0])
                + math.cos(self.angle) * (-f1[1] + f2[1])
            )
        )
        va = np.clip(va, self.min_velocity_angle, self.max_velocity_angle)
        return vx, vy, va

    def velocity(self, ax, ay, a_angle, dt: float):
        vx = self.x_speed + ax * dt
        vy = self.y_speed + ay * dt
        va = self.angle_speed + a_angle * dt
        debug_print("velocity", vx, vy, va)
        return vx, vy, va

    # def update_with_wrench(self, action, delta_t):
    #     a_x, a_y, a_angle = action
    #     self.x = self.x + self.x_speed * delta_t + 0.5 * a_x * delta_t * delta_t
    #     self.y = self.y + self.y_speed * delta_t + 0.5 * a_y * delta_t * delta_t
    #     self.angle = (
    #         self.angle + self.angle_speed * delta_t + 0.5 * a_angle * delta_t * delta_t
    #     )
    #     self.angle = self.angle % (2 * np.pi)
    #     self.x_speed, self.y_speed, self.angle_speed = self.velocity(
    #         a_x, a_y, a_angle, delta_t
    #     )
    #     return np.array([self.x, self.y, self.angle])

    def update(
        self,
        action: np.ndarray,
        delta_t: float,
        update_image=True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Internal updates for the table.

        Parameters
        ----------
        action : np.ndarray, shape=(2)
            Action taken by the RL agent.
        delta_t : float
            Temporal resolution of the dynamics.
        step : int
            Time step for blind secondary agent (used for applying open-loop policy).
        interact_mode : bool
            Flag indicating whether table is in interact mode.

        Returns
        -------
        f1 : np.ndarray, shape=(2)
            Actions for the non-RL synthetic agent.
        f2 : np.ndarray, shape=(2)
            Actions for the RL agent.
        x : float
            X position of the RL agent.
        y : float
            Y position of the RL agent.
        """
        # take action (RL agent)
        action_clipped = np.clip(action, -self.cap, self.cap)
        debug_print("ACTION clipped", action_clipped.shape, action_clipped)
        player_1_act = action_clipped[0, :]
        player_2_act = action_clipped[1, :]
        # f1: for other agent: option to use synthetic data or take realtime inputs via mouse
        # take mouse position as human force
        """
        if interact_mode and pygame.mouse.get_pressed()[0]:
            dx, dy = pygame.mouse.get_pos()

            fx = dx - self.rect.x - self.rect.size[0] / 2
            fy = dy - self.rect.y - self.rect.size[1] / 2

            # place a cap on human force input (no super human strength)
            fy = np.clip(fy, -self.cap, self.cap)

        else:
            # f1: feed synthetic data, scale the inputs to convert to pixels
            self.f1 = np.array([self.curr_traj[step][0], self.curr_traj[step][1]])"""

        f2s = np.array(player_2_act) * self.policy_scaling_factor
        f1s = np.array(player_1_act) * self.policy_scaling_factor

        # store previous step's values
        self.px = self.x
        self.py = self.y
        self.pangle = self.angle

        # self.x_speed += a_x * delta_t
        # self.y_speed += a_y * delta_t
        # self.angle_speed += a_angle * delta_t
        # debug_print("xspd, dt:", self.x_speed, delta_t)  # DEBUG
        if self.physics_control_type == "force":
            # convert these inputs to the table acceleration
            a_x, a_y, a_angle = self.acceleration(f1s, f2s)
            # integrate to get position
            # self.x += self.x_speed * delta_t
            # self.y += self.y_speed * delta_t
            # self.angle += self.angle_speed * delta_t
            self.x = self.x + self.x_speed * delta_t + 0.5 * a_x * delta_t * delta_t
            self.y = self.y + self.y_speed * delta_t + 0.5 * a_y * delta_t * delta_t
            self.angle = (
                self.angle
                + self.angle_speed * delta_t
                + 0.5 * a_angle * delta_t * delta_t
            )
            self.angle = self.angle % (2 * np.pi)
            self.x_speed, self.y_speed, self.angle_speed = self.velocity(
                a_x, a_y, a_angle, delta_t
            )
        elif self.physics_control_type == "velocity":
            self.x_speed, self.y_speed, self.angle_speed = self.control_to_velocity(
                f1s, f2s
            )
            self.x = self.x + self.x_speed * delta_t
            self.y = self.y + self.y_speed * delta_t
            self.angle = self.angle + self.angle_speed * delta_t
        debug_print("Updated x, y, angle", self.x, self.y, self.angle)

        self.table_center_to_player1 = np.array([self.x + self.length_from_center_to_person * np.cos(self.angle), 
                        self.y + self.length_from_center_to_person * np.sin(self.angle)])
        self.table_center_to_player2 = np.array([self.x - self.length_from_center_to_person * np.cos(self.angle),
                        self.y - self.length_from_center_to_person * np.sin(self.angle)])
        # update the table position
        # TODO: this formula is wrong, needs to take rotation into account
        # rot_img = pygame.transform.rotate(
        #   self.image, math.degrees(self.angle))
        # self.rect = self.image.get_rect(center=self.image.get_rect(center=(self.x, self.y)).center)
        # testx = self.x - self.rect.size[0] / 2
        # self.rect.x = self.x - self.rect.size[0] / 2
        # self.rect.y = self.y - self.rect.size[1] / 2

        # offset from pivot to center
        # pos = (self.x, self.y)
        angle = math.degrees(self.angle)
        # originPos = (self.w/2, self.h/2)
        # image_rect = self.original_img.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
        # offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
        # roatated offset from pivot to center
        # rotated_offset = offset_center_to_pivot.rotate(-angle)
        # roatetd image center
        # rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)
        # get a rotated image
        if update_image:
            self.image = pygame.transform.rotate(self.original_img, angle)
            # self.rect = self.image.get_rect(center = rotated_image_center)
            # self.rect.center = rotated_image_center
            self.rect = self.image.get_rect(center=(self.x, self.y))
            # self.rect = self.image.get_rect(center = self.rect.center)
            self.mask = pygame.mask.from_surface(self.image)

        # self.image = pygame.transform.rotate(self.original_img, math.degrees(self.angle))
        # self.rect = self.image.get_rect(center = self.image.get_rect(center=(self.x, self.y)).center) #self.rect.center)
        # debug_print('table update', self.x, self.rect.x, testx, self.y, self.rect.y, self.angle, f1s, f2s)
        return (
            f1s,
            f2s,
            self.x,
            self.y,
            self.angle,
            self.x_speed,
            self.y_speed,
            self.angle_speed,
        )


class Agent(object):
    def __init__(self) -> None:
        super(Agent, self).__init__()
        # Agents have fx and fy
        self.f = np.array([0, 0])
        # Limit the action magnitude
        self.cap = 1.0
        self.velocity_cap = 10.0
        # TODO: for pygame what is the action scaling factor? play around with arrow keys
        self.policy_scaling_factor = 200.0
        # scripted actions
        self.action_callback = None
