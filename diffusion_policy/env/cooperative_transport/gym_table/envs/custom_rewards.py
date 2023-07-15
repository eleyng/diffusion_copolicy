import numpy as np

from diffusion_policy.env.cooperative_transport.gym_table.envs.utils import CONST_DT, WINDOW_H, WINDOW_W, L
# from libs.planner.planner_utils import pid_single_step, is_safe


## Define custom reward functions here
def custom_reward_function(states, goal, obs, env=None, vectorized=False, interaction_forces=False, skip=5, u_r=None, u_h=None, collision=None, collision_checking_env=None, success=None):
    # states should be an N x state_dim array
    assert (
        len(states.shape) == 2
    ), "state shape mismatch for compute_reward. Expected (n, {0}), where n is the set of states you are evaluating. Got {1}".format(
        states.shape
    )

    assert states is not None, "states parameter cannot be None"

    n = states.shape[0]
    reward = np.zeros(n)
    # slack reward
    reward += -0.1

    # if collision is None or success is None:
    #     assert env is not None, "env parameter cannot be None if collision is not provided"
    #     angle = np.arctan2(states[:, 3], states[:, 2])
    #     check_states = np.stack(np.array([states[:, 0], states[:, 1], angle]), axis=1)
    #     collision_and_success = np.array([env.mp_check_collision_and_success(check_states[i, :]) for i in range(n)])
    #     for check in range(n):
    #         reward[check] += -100.0 if collision_and_success[check, 0] else 0.0
    #         reward[check] += 100.0 if collision_and_success[check, 1] else 0.0
    # else:
    #     if collision:
    #         reward += -100.0
    #     if success:
    #         reward += 100.0
    # r_collision = np.zeros(n)
    # if collision_checking_env is not None:
    #     for i in range(n):
    #         r_collision[i] = -100.0 if not is_safe(states[i, :], collision_checking_env=collision_checking_env) else 0.0
    # reward += r_collision


    dg = np.linalg.norm(states[:, :2] - goal, axis=1)

    sigma_g = 300
    r_g = np.exp(-np.power(dg, 2) / (2 * sigma_g ** 2))
    reward += r_g

    r_obs = np.zeros(n)
    sigma_o = 50

    num_obstacles = obs.shape[0]
    if states is not None:
        d2obs_lst = np.asarray(
            [
                np.linalg.norm(states[:, :2] - obs[i, :], axis=1)
                for i in range(num_obstacles)
            ],
            dtype=np.float32,
        )

    # negative rewards for getting close to wall
    for i in range(num_obstacles):
        d = d2obs_lst[i]
        r_obs += - np.exp(-np.power(d, 2) / (2 * sigma_o ** 2))

    r_obs += - np.exp(-np.power((states[:, 0] - 0), 2) / (2 * sigma_o ** 2))
    r_obs += - np.exp(-np.power((states[:, 0] - WINDOW_W), 2) / (2 * sigma_o ** 2))
    r_obs += - np.exp(-np.power((states[:, 1] - 0), 2) / (2 * sigma_o ** 2))
    r_obs += - np.exp(-np.power((states[:, 1] - WINDOW_H), 2) / (2 * sigma_o ** 2))

    reward += r_obs

    if interaction_forces:
        if states.shape[0] == 1:
            interaction_forces = compute_interaction_forces(states[0, :4], u_r, u_h)
        else:
            pid_actions = pid_single_step(
                                env,
                                states[skip, :4],
                                kp=0.15,
                                ki=0.0,
                                kd=0.0,
                                max_iter=40,
                                dt=CONST_DT,
                                eps=1e-2,
                                u_h=u_h.squeeze().numpy(),
                            )
            pid_actions /= np.linalg.norm(pid_actions)
            interaction_forces = compute_interaction_forces(states[skip, :4], pid_actions, u_h.detach().numpy().squeeze())
        reward += interaction_forces_reward(interaction_forces)

    if not vectorized:
        return reward[0]
    else:
        return reward


def interaction_forces_reward(interaction_forces):
    # interaction forces penalty : penalize as interaction forces stray from zero
    penalty = 0.5 * (interaction_forces / 1000 ** 2)

    return -penalty

def compute_interaction_forces(table_state, f1, f2):
    table_center_to_player1 = np.array(
            [
                table_state[0] + (L/2) * table_state[2],
                table_state[1] + (L/2) * table_state[3],
            ]
        )
    table_center_to_player2 = np.array(
        [
            table_state[0] - (L/2) * table_state[2],
            table_state[1] - (L/2) * table_state[3],
        ]
    )
    inter_f = (f1 - f2) @ (
            table_center_to_player1 - table_center_to_player2
    )
    return inter_f
