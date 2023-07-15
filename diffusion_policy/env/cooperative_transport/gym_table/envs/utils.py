import math

import numpy as np
import pygame
import yaml

VERBOSE = False  # Set to True to print debug info

# game loop parameters
FPS = 30  # frames per second max limited to this when rendering in game loop
CONST_DT = 1 / FPS  # constant time step for physics
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS

# pygame physics parameters
m = 2.0  # mass of the table
b = 2.0  # linear velocity damping coefficient
I = 1.0  # moment of inertia of the table
L = 1.0  # length of the table
d = 40  # damping parameter related to angular velocity

# sprite parameters
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHTBLUE = (138, 181, 255)
BLUE = (20, 104, 250)
LIGHTORANGE = (255, 225, 173)
ORANGE = (255, 167, 15)
GREEN = (100, 183, 105)
WINDOW_W = 1200
WINDOW_H = 600
STATE_W = 256
STATE_H = 256
rad = np.pi / 180.0
wallpts = np.array([[0, 0], [0, WINDOW_H], [WINDOW_W, WINDOW_H], [WINDOW_W, 0], [0, 0]])
obstacle_size = 45



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


def load_cfg(config_file):
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def consec_repeat_starts(a, n):
    '''Returns the indices of the first element of each repeat of length n in a
    
        Parameters
        ----------
        a: 1D array
        n: int, length of repeat

    '''

    N = n - 1
    m = a[:-1] == a[1:]

    return np.flatnonzero(np.convolve(m, np.ones(N, dtype=int)) == N) - N + 1


def get_idx_repeats_of_len_n(states_xy, n):
    '''Returns 2D array of the intersection of indices of each repeat of length n
        in the x and y coordinates of states_xy. 
    
        Parameters
        ----------
        states_xy: 2D array, shape (n, 2)
        n: int, length of repeat
    
    '''
    repeats_x = consec_repeat_starts(states_xy[:, 0], n)
    repeats_y = consec_repeat_starts(states_xy[:, 1], n)
    repeat_intersect = np.intersect1d(repeats_x, repeats_y)

    return repeat_intersect


def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def rect_distance(rect1, rect2):
    """
    Reference: https://stackoverflow.com/a/3040439
    https://stackoverflow.com/a/58781995
    """
    # if in collision then dist is 0.0
    if rect1.colliderect(rect2):
        return 0.0

    rect1_vertices = np.array(
        [rect1.topleft, rect1.topright, rect1.bottomright, rect1.bottomleft]
    )
    rect2_vertices = np.array(
        [rect2.topleft, rect2.topright, rect2.bottomright, rect2.bottomleft]
    )

    # vertex distance
    vertex_dist = float("inf")
    for i in range(4):
        for j in range(4):
            vertex_dist = min(
                vertex_dist, np.linalg.norm(rect1_vertices[i] - rect2_vertices[j])
            )

    # points from 1, segments from 2
    pt_seg_dist_1 = float("inf")
    seg_a = rect2_vertices
    seg_b = np.vstack([rect2_vertices[1:], rect2_vertices[0]])
    # print(seg_a, seg_b)
    for i in range(4):
        pt_seg_dist_1 = min(
            pt_seg_dist_1, np.min(lineseg_dists(rect1_vertices[i], seg_a, seg_b))
        )

    # points from 2, segments from 1
    pt_seg_dist_2 = float("inf")
    seg_a = rect1_vertices
    seg_b = np.vstack([rect1_vertices[1:], rect1_vertices[0]])
    # print(seg_a, seg_b)
    for i in range(4):
        pt_seg_dist_1 = min(
            pt_seg_dist_2, np.min(lineseg_dists(rect2_vertices[i], seg_a, seg_b))
        )
    min_dist = min(vertex_dist, min(pt_seg_dist_1, pt_seg_dist_2))
    # print("Min dist: ", min_dist)
    return min_dist

    # the rest of the code are used for checking axis-aligned rectangle distances only
    left = x2b < x1
    right = x1b < x2
    top = y2b < y1
    bottom = y1b < y2
    if bottom and left:
        print("bottom left")
        return math.hypot(x2b - x1, y2 - y1b)
    elif left and top:
        print("top left")
        return math.hypot(x2b - x1, y2b - y1)
    elif top and right:
        print("top right")
        return math.hypot(x2 - x1b, y2b - y1)
    elif right and bottom:
        print("bottom right")
        return math.hypot(x2 - x1b, y2 - y1b)
    elif left:
        print("left")
        return x1 - x2b
    elif right:
        print("right")
        return x2 - x1b
    elif top:
        print("top")
        return y1 - y2b
    elif bottom:
        print("bottom")
        return y2 - y1b
    else:  # rectangles intersect
        print("intersection")
        return 0.0


def debug_print(*args):
    if not VERBOSE:
        return
    print(*args)


def set_action_keyboard(action):
    # debug_print('set action: ', action)
    u = np.zeros((2, 2))
    # player 1 acts only
    if action == 1:
        u[0, 0] = +1.0  # x-dim, right
    if action == 2:
        u[0, 0] = -1.0  # x-dim, left
    if action == 3:
        u[0, 1] = -1.0  # y-dim, up
    if action == 4:
        u[0, 1] = +1.0  # y-dim, down
    # player 2 acts only
    if action == 5:
        u[1, 0] = +1.0  # x-dim, right
    if action == 6:
        u[1, 0] = -1.0  # x-dim, left
    if action == 7:
        u[1, 1] = -1.0  # y-dim, up
    if action == 8:
        u[1, 1] = +1.0  # y-dim, down
    # player 1 / 2 mixed action
    if action == 9:
        u[:, 0] = +1.0  # x-dim, right
    if action == 10:
        u[0, 0] = +1.0  # x-dim, right
        u[1, 0] = -1.0  # x-dim, left
    if action == 11:
        u[0, 0] = +1.0  # x-dim, right
        u[1, 1] = -1.0  # y-dim, up
    if action == 12:
        u[0, 0] = +1.0  # x-dim, right
        u[1, 1] = +1.0  # y-dim, down

    if action == 13:
        u[0, 0] = -1.0  # x-dim, left
        u[1, 0] = +1.0  # x-dim, right
    if action == 14:
        u[:, 0] = -1.0  # x-dim, left
    if action == 15:
        u[0, 0] = -1.0  # x-dim, left
        u[1, 1] = -1.0  # y-dim, up
    if action == 16:
        u[0, 0] = -1.0  # x-dim, left
        u[1, 1] = +1.0  # y-dim, down

    if action == 17:
        u[0, 1] = -1.0  # y-dim, up
        u[1, 0] = +1.0  # x-dim, right
    if action == 18:
        u[0, 1] = -1.0  # y-dim, up
        u[1, 0] = -1.0  # x-dim, left
    if action == 19:
        u[:, 1] = -1.0  # y-dim, up
    if action == 20:
        u[0, 1] = -1.0  # y-dim, up
        u[1, 1] = +1.0  # y-dim, down

    if action == 21:
        u[0, 1] = +1.0  # y-dim, down
        u[1, 0] = +1.0  # x-dim, right
    if action == 22:
        u[0, 1] = +1.0  # y-dim, down
        u[1, 0] = -1.0  # x-dim, left
    if action == 23:
        u[0, 1] = +1.0  # y-dim, down
        u[1, 1] = -1.0  # y-dim, up
    if action == 24:
        u[:, 1] = +1.0  # y-dim, down
    return u


def set_action_joystick(action):
    u = np.zeros((2, 2))

    # player 1 acts
    u[0, 0] = action[0]
    u[0, 1] = action[1]
    # player 2 acts only
    u[1, 0] = action[2]
    u[1, 1] = action[3]
    return u


def get_keys_to_action():
    """For discrete actions. Returns mapping from keys pressed to action performed."""
    keys_to_action = {}
    keys_to_action[()] = 0  # no act
    # player 2 actions only - CIRCLE
    keys_to_action[(ord("d"),)] = 5  # right
    keys_to_action[(ord("a"),)] = 6  # left
    keys_to_action[(ord("w"),)] = 7  # up
    keys_to_action[(ord("s"),)] = 8  # down
    # player 1 actions only - TRIANGLE
    keys_to_action[(pygame.K_RIGHT,)] = 1  # or 1 if doing multidiscrete
    keys_to_action[(pygame.K_LEFT,)] = 2
    keys_to_action[(pygame.K_UP,)] = 3
    keys_to_action[(pygame.K_DOWN,)] = 4
    # player 1 / 2 mixed action
    keys_to_action[(ord("d"), pygame.K_RIGHT)] = 9  # or 1 if doing multidiscrete
    keys_to_action[(ord("a"), pygame.K_RIGHT)] = 10  # or 1 if doing multidiscrete
    keys_to_action[(ord("w"), pygame.K_RIGHT)] = 11  # or 1 if doing multidiscrete
    keys_to_action[(ord("s"), pygame.K_RIGHT)] = 12  # or 1 if doing multidiscrete

    keys_to_action[(ord("d"), pygame.K_LEFT)] = 13
    keys_to_action[(ord("a"), pygame.K_LEFT)] = 14
    keys_to_action[(ord("w"), pygame.K_LEFT)] = 15
    keys_to_action[(ord("s"), pygame.K_LEFT)] = 16

    keys_to_action[(ord("d"), pygame.K_UP)] = 17
    keys_to_action[(ord("a"), pygame.K_UP)] = 18
    keys_to_action[(ord("w"), pygame.K_UP)] = 19
    keys_to_action[(ord("s"), pygame.K_UP)] = 20

    keys_to_action[(ord("d"), pygame.K_DOWN)] = 21
    keys_to_action[(ord("a"), pygame.K_DOWN)] = 22
    keys_to_action[(ord("w"), pygame.K_DOWN)] = 23
    keys_to_action[(ord("s"), pygame.K_DOWN)] = 24

    return keys_to_action
