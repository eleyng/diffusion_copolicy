import pdb
import random
import argparse
import torch
import yaml
import gym
import time
import copy
import pygame
import numpy as np
from numpy.linalg import norm
from os.path import join, exists, dirname, abspath
from os import mkdir
from stable_baselines3.common.env_checker import check_env
from play import init_joystick
from plot_with_data import plot_mse_state
import sys

sys.path.insert(1, dirname(abspath(__file__)) + "/../models")
from bc import sample_discrete, BCRNN
from mdrnn import MDRNN

sys.path.insert(1, dirname(abspath(__file__)) + "/../envs")
from utils import load_cfg

FPS = 30
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS


def compute_reward(env, state_table):  ## TODO: update to actual reward function

    state_table = env.unstandardize(state_table, mean_s[:2], std_s[:2]).detach().numpy()
    target = np.array([env.target.x, env.target.y])

    r = np.zeros((K, 1))

    # reward for shorter time (small enough to not incentivize obstacle collision)
    r = np.add(r, -0.1)

    # reward for distance to goal
    dg = norm(np.subtract(target, state_table), axis=-1)
    a = 0.98
    const = 100.0
    r_g = 10.0 * np.power(a, dg - const)
    r = np.add(r, r_g)

    # reward for distance to obs
    obstacles = env.obstacles
    num_obs = obstacles.shape[0]
    b = -8.0
    c = 0.9
    const = 130.0
    const_w = 100.0
    for obs in obstacles:
        d = norm((obs - state_table), axis=-1)
        r = np.add(r, b * np.power(c, d - const))

    # reward for distance to wall
    for i in range(len(env.wallpts) - 1):
        wall_start = np.array(env.wallpts[i])
        wall_end = np.array(env.wallpts[i + 1])
        dw = norm(
            np.cross(wall_start - wall_end, wall_end - state_table), axis=-1
        ) / norm(wall_start - wall_end)
        r = np.add(r, b * np.power(c, np.expand_dims(dw, axis=-1) - const_w))

    # print("Total step reward: ", r)
    return r


torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
yaml_filepath = join(dirname(__file__), "../config/inference_params.yml")
parser = argparse.ArgumentParser("Model training & validation")
parser.add_argument(
    "--logdir", type=str, help="Where things are logged and models are loaded from."
)
parser.add_argument(
    "--noreload", action="store_true", help="Do not reload if specified."
)
args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("device: ", device)

# Load parameters from config file
cfg = load_cfg(yaml_filepath)
control_type = cfg["control_type"]
ep = cfg["ep"]
data_base = cfg["data_base"]
mode = cfg["mode"]
run_mode = cfg["run_mode"]
strategy_name = cfg["strategy_name"]
map_cfg = cfg["map_config"]
print(map_cfg)
experiment_name = cfg["experiment_name"]
exp_string = cfg["exp_string"]
print(experiment_name)
train_stats_f = cfg["train_stats_f"]
data_dir = cfg["data_dir"]
checks = cfg["checks"]
n_layers = cfg["n_layers"]
n_classes = cfg["n_classes"]
LSIZE = cfg["LSIZE"]
RSIZE = cfg["RSIZE"]
print("RSIZE", RSIZE)
ASIZE = cfg["ASIZE"]
BSIZE = cfg["BSIZE"]
NGAUSS = cfg["NGAUSS"]
print("NGAUSS", NGAUSS)
AGAUSS = cfg["AGAUSS"]
print("AGAUSS", AGAUSS)
SEQ_LEN = cfg["SEQ_LEN"]
print("SEQ_LEN", SEQ_LEN)
H = cfg["H"]
print("H", H)
fig_border = cfg["fig_border"]
teacher_forcing_ratio = cfg["teacher_forcing"]
freq = cfg["freq"]
epochs = cfg["epochs"]
time_limit = cfg["time_limit"]
beta_min = cfg["beta_min"]
beta_max = cfg["beta_max"]
map_cfg = cfg["map_config"]

# map_cfg = load_cfg(join("../config/maps", cfg["map_config"]))
# print(map_cfg)


# Initialize other parameters
ratio = 0.0
K = NGAUSS * AGAUSS

# Load human data if mode == "recorded_data"
if mode == "recorded_data":
    data = dict(
        np.load(
            join(
                dirname(__file__),
                "../envs/" + data_dir + "/" + data_dir.split("/")[-1] + ".npz",
            )
        )
    )
    print("Trajectory length: ", len(data["states"]))
    data_length = len(data["states"])
elif mode == "human_player":
    joysticks = init_joystick()

# Get training data statistics for standardizations
train_stats = dict(
    np.load(
        join(
            dirname(__file__),
            "../envs/" + data_dir.split("/")[-4] + "/" + train_stats_f,
        )
    )
)
mean_s = torch.Tensor(train_stats["mean_s"]).to(device, non_blocking=True)
std_s = torch.Tensor(train_stats["std_s"]).to(device, non_blocking=True)
mean_ns = torch.Tensor(train_stats["mean_ns"]).to(device, non_blocking=True)
std_ns = torch.Tensor(train_stats["std_ns"]).to(device, non_blocking=True)


# Loading model
model_dir = join(dirname(__file__), "../", data_base, experiment_name)
model_file = join(model_dir, "best.tar")
if not exists(model_dir):
    mkdir(model_dir)
# Create samples directory
sample_dir = join(dirname(__file__), "../", data_base, model_dir, "samples")
if not exists(sample_dir):
    mkdir(sample_dir)

# Initialize dynamical model
model = MDRNN(
    LSIZE,
    ASIZE,
    RSIZE,
    NGAUSS,
    AGAUSS,
    BSIZE,
    n_layers,
    device,
    n_classes,
    control_type,
).float()
model.to(device, non_blocking=True)

# Initialize training parameters
optimizer = torch.optim.RMSprop(
    model.parameters(), lr=1e-4, alpha=0.9, weight_decay=0.2
)

# Load dynamical model parameters
if exists(model_file) and not args.noreload:
    model_state = torch.load(
        model_file
    )  # , map_location=torch.device('cpu')) if device == 'cpu' else torch.load(vae_file)
    print(
        "Loading model at epoch {} "
        "with test error {}".format(model_state["epoch"], model_state["precision"])
    )
    state_dict = model_state["state_dict"]
    state_dict2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if "_l0" in key:
            name = key.split("_")
            state_dict2["_".join(name[:-1])] = state_dict2.pop(key)
    if mode == "rollout":
        model.load_state_dict(state_dict2)
    else:
        model.load_state_dict(state_dict)
    optimizer.load_state_dict(model_state["optimizer"])
    # scheduler.load_state_dict(model_state["scheduler"])
    # earlystopping.load_state_dict(model_state['earlystopping'])


# Starting env
map_cfg_filepath = join(dirname(__file__), "../config/maps", map_cfg)
print("map_cfg_filepath", map_cfg_filepath)
env = gym.make(
    "cooperative_transport.gym_table:table-v0",
    control="joystick",
    map_config=map_cfg_filepath,
    run_mode=run_mode,
    strategy_name=strategy_name,
    ep=ep,
)
env_r = gym.make(
    "cooperative_transport.gym_table:table-v0",
    control="joystick",
    map_config=map_cfg_filepath,
    run_mode=run_mode,
    strategy_name=strategy_name,
    ep=ep,
)
# folder for saving prediction plots
plt_pred_dir = join(env.dirname_vis, "pred_plots")
if not exists(plt_pred_dir):
    mkdir(plt_pred_dir)

# obs = env.reset()
# Standardize obs
# print(obs.shape, mean_s.shape, std_s.shape)
# obs = env.standardize(torch.tensor(obs), mean_s, std_s)
n_steps = 100001
n_iter = 0
running = True
env_done = False
next_game_tick = time.time()
clock = pygame.time.Clock()

u_lst = torch.zeros((1, ASIZE)).to(device, non_blocking=True)
obs_lst = torch.zeros((1, LSIZE)).to(device, non_blocking=True)

hidden_a = torch.zeros((n_layers, BSIZE, RSIZE)).to(device, non_blocking=True)
hidden_s = torch.zeros((n_layers, BSIZE, RSIZE)).to(device, non_blocking=True)
hidden_a_lst = torch.zeros((H, n_layers, BSIZE, RSIZE)).to(device, non_blocking=True)
hidden_s_lst = torch.zeros((H, n_layers, BSIZE, RSIZE)).to(device, non_blocking=True)

initial_observations = []  # .to(device, non_blocking=True)
initial_actions = []  # .to(device, non_blocking=True)
rollout_observations = []  # .to(device, non_blocking=True)
rollout_actions = []  # .to(device, non_blocking=True)

while running:
    loops = 0
    print("Done?", env.done)
    if env.done:
        print("Done.")
        pygame.quit()
        break
    else:
        while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not env.done:
            # Get human input (last two inputs)
            if mode == "recorded_data" and env.n_step < data["actions"].shape[0]:
                u_h = data["actions"][env.n_step, 2:]
                u_h_d = data["actions"][env.n_step, :]
                u_h[0] = u_h_d[-2]
                u_h[1] = u_h_d[-1]
                # u_h[2] = u_h_d[2]
                # u_h[-1] = u_h_d[-1]

                print("Data in MPC: ", u_h, H)
            elif mode == "human_player":
                p1_id = 0
                u_h = np.array(
                    [
                        joysticks[p1_id].get_axis(0),
                        joysticks[p1_id].get_axis(1),
                    ]
                )

            with torch.no_grad():

                # n_iter < H: collecting initial observations, robot does not act
                if n_iter < H:
                    print("*****USING DATA*****", n_iter, H)
                    # Create action tensor
                    u_tmp = np.zeros((1, 1, ASIZE))
                    u_tmp[:, :, :] = u_h_d
                    u_tmp = torch.tensor(u_tmp).float().to(device)

                    # Choose robot actions
                    # u_tmp[:, :, 0] = 0.5 * torch.ones((1, 1, 1))  # 0.5, right
                    # u_tmp[:, :, 0] = 1.0 * torch.ones((1, 1, 1))  # 1.0, right
                    # u_tmp[:, :, 0] = torch.randn((1, 1, 2))  # rand x ,y
                    # comment out all for no action

                    obs, reward, done, info = env.step(u_tmp.squeeze(), CONST_DT)
                    print(mean_s.shape, obs.shape)
                    obs = env.standardize(torch.tensor(obs), mean_s, std_s)

                    # Create obs tensor
                    obs_t = obs.unsqueeze(0).unsqueeze(0).float().to(device)

                    # Update hidden state only
                    ins_s = obs_t
                    ins_a = u_tmp

                    _, _, _, _, _, _, hidden_s, hidden_a = model(
                        actions=ins_a, states=ins_s, h=hidden_s, h_a=hidden_a
                    )

                    hidden_s_lst[n_iter, ...] = hidden_s
                    hidden_a_lst[n_iter, ...] = hidden_a

                # n_iter >= H: robot plan & act
                else:
                    # pdb.set_trace()
                    print("*****PREDICTING*****")
                    # Rollout  futures up to length SEQ_LEN
                    pred_act = torch.zeros(
                        (BSIZE, H + SEQ_LEN, NGAUSS * AGAUSS, ASIZE)
                    ).to(device, non_blocking=True)
                    sig_a_lst = torch.zeros(
                        (BSIZE, H + SEQ_LEN, NGAUSS * AGAUSS, ASIZE)
                    ).to(device, non_blocking=True)
                    pred_s = torch.zeros(
                        (BSIZE, H + SEQ_LEN, NGAUSS * AGAUSS, LSIZE)
                    ).to(device, non_blocking=True)
                    sig_s_lst = torch.zeros(
                        (BSIZE, H + SEQ_LEN, NGAUSS * AGAUSS, LSIZE)
                    ).to(device, non_blocking=True)
                    pi_lst = torch.zeros((BSIZE, 1, NGAUSS)).to(
                        device, non_blocking=True
                    )
                    pi_a_lst = torch.zeros((BSIZE, 1, AGAUSS)).to(
                        device, non_blocking=True
                    )

                    cum_r = np.zeros((K, 1))  # B x T
                    print(u_lst[n_iter - H + 1 :, :].unsqueeze(0).shape)
                    print(obs_lst[n_iter - H + 1 :, :].unsqueeze(0).shape)
                    # UPDATE true hidden state
                    (
                        mus,
                        sigmas,
                        logpis,
                        mu_a,
                        sig_a,
                        logpi_a,
                        hidden_tf,
                        hidden_tf_a,
                    ) = model(
                        actions=u_lst[n_iter - H + 1 :, :].unsqueeze(0),
                        states=obs_lst[n_iter - H + 1 :, :].unsqueeze(0),
                        h=hidden_s_lst[n_iter - H],
                        h_a=hidden_a_lst[n_iter - H],
                    )

                    for n in range(NGAUSS):
                        ns, _ = model.get_gmm_sample_from_mode(
                            mus.detach().clone(),
                            sigmas.detach().clone(),
                            n,
                            False,
                        )
                        # pdb.set_trace()
                        for m in range(AGAUSS):
                            sig = sigmas[:, :, n, :]
                            # pdb.set_trace()
                            sig_s_lst[:, :H, (NGAUSS * m) + n, :] = sig
                            pred_s[:, :H, (NGAUSS * m) + n, :] = ns

                            na, _ = model.get_gmm_sample_from_mode(
                                mu_a.detach().clone(),
                                sig_a.detach().clone(),
                                m,
                                False,
                            )
                            sig = sig_a[:, :, m, :].squeeze()
                            sig_a_lst[:, :H, (NGAUSS * m) + n, :] = sig
                            pred_act[:, :H, (NGAUSS * m) + n, :] = na

                    hidden_s = hidden_tf.repeat(1, NGAUSS * AGAUSS, 1)
                    hidden_a = hidden_tf_a.repeat(1, NGAUSS * AGAUSS, 1)
                    # pdb.set_trace()
                    ins_a = u_lst[-1, :].repeat(NGAUSS * AGAUSS, 1, 1)
                    ins_s = obs_lst[-1, :].repeat(NGAUSS * AGAUSS, 1, 1)
                    for t in range(n_iter, n_iter + SEQ_LEN):

                        (
                            mus,
                            sigmas,
                            logpis,
                            mu_a,
                            sig_a,
                            logpi_a,
                            hidden_tf,
                            hidden_tf_a,
                        ) = model(actions=ins_a, states=ins_s, h=hidden_s, h_a=hidden_a)
                        if t == n_iter:
                            pi_lst = torch.exp(logpis.detach().clone())
                            pi_a_lst = torch.exp(logpi_a.detach().clone())
                        for n in range(NGAUSS):
                            # print("n", n)
                            for m in range(AGAUSS):
                                # print(
                                #     "m", m, sigmas[(NGAUSS * m) + n].unsqueeze(0).shape
                                # )
                                ns, _ = model.get_gmm_sample_from_mode(
                                    mus[(NGAUSS * m) + n].unsqueeze(0).detach().clone(),
                                    sigmas[(NGAUSS * m) + n]
                                    .unsqueeze(0)
                                    .detach()
                                    .clone(),
                                    n,
                                    False,
                                )  # mus should be B, T,G,  F (1, 1, 8, 14)
                                # print(ns.shape, pred_s.shape)
                                # print(sigmas.shape)
                                # print("ns", ns.shape, pred_s.shape)
                                pred_s[
                                    :, H + (t - n_iter), (NGAUSS * m) + n, :
                                ] = ns  # .squeeze(1)
                                # print(
                                #     sig_s_lst[:, t - n_iter, (NGAUSS * m) + n, :].shape
                                # )
                                # print("t", t)
                                # print(sigmas[(NGAUSS * m) + n, :, n, :].shape)
                                sig_s_lst[
                                    :, H + (t - n_iter), (NGAUSS * m) + n, :
                                ] = sigmas[(NGAUSS * m) + n, :, n, :]

                                na, _ = model.get_gmm_sample_from_mode(
                                    mu_a[(NGAUSS * m) + n]
                                    .unsqueeze(0)
                                    .detach()
                                    .clone(),
                                    sig_a[(NGAUSS * m) + n]
                                    .unsqueeze(0)
                                    .detach()
                                    .clone(),
                                    m,
                                    False,
                                )  # mus should be B, T,G,  F (1, 1, 2, 14)
                                # print("na", na.shape, mu_a.shape)
                                pred_act[:, H + (t - n_iter), (NGAUSS * m) + n, :] = na
                                sig_a_lst[
                                    :, H + (t - n_iter), (NGAUSS * m) + n, :
                                ] = sig_a[(NGAUSS * m) + n, :, m, :]
                        # print("pred act", pred_act.shape)
                        ins_a = pred_act[:, H + (t - n_iter), :, :].permute(
                            1, 0, 2
                        )  # B, T, 1, F
                        ins_s = pred_s[:, H + (t - n_iter), :, :].permute(1, 0, 2)
                        # print("pred ins", ins_s.shape)

                        if t == n_iter:
                            tmp_hidden_s = hidden_tf
                            tmp_hidden_a = hidden_tf_a

                        # ins_a = pred_act[:, t, :].unsqueeze(1)
                        # ins_s = pred_s[:, t, :].unsqueeze(1)
                    # pdb.set_trace()

                    # Compute reward using state
                    # rs = np.zeros((K,))
                    # for k in range(K):
                    #     rs_k = 0
                    #     for t in range(H + SEQ_LEN):
                    #         # spdb.set_trace()
                    #         rs_k = env_r.compute_reward(
                    #             state=pred_s.squeeze()[t, k, :3].detach().numpy()
                    #         )
                    #         # pdb.set_trace()
                    #     rs[k] = rs_k
                    #     # pdb.set_trace()

                    # cum_r = rs  # np.add(cum_r, rs)

                    pred_s_unstd = env.unstandardize(pred_s, mean_ns, std_ns).to(
                        device, non_blocking=True
                    )
                    # pdb.set_trace()
                    if n_iter % 10 == 0:
                        # pdb.set_trace()
                        plot_mse_state(
                            pred_s_unstd.contiguous().cpu().numpy(),
                            epoch=env.n_step,
                            save_dir=plt_pred_dir,
                            std=sig_s_lst.detach().clone().cpu().numpy(),
                            start=env.unstandardize(obs_t, mean_s, std_s)
                            .detach()
                            .numpy(),
                        )

                    # Select best action
                    # print("Reward", cum_r)
                    # best_idx = np.where(cum_r.squeeze() == cum_r.max())
                    # print("Best idx", best_idx[0][0], u_tmp.shape)
                    # pdb.set_trace()
                    best_idx_s = np.argmax(pi_lst.squeeze().clone().cpu().numpy())
                    best_idx_a = np.argmax(pi_a_lst.squeeze().clone().cpu().numpy())
                    best_idx = NGAUSS * best_idx_a + best_idx_s

                    # u_tmp = pred_act[:, H, best_idx[0][0], :]  # .detach().numpy()
                    u_tmp = pred_act[:, H, best_idx, :]  # .detach().numpy()
                    u_tmp = torch.tensor(u_tmp).float().to(device)

                    # pdb.set_trace()
                    # u_lst = torch.cat((u_lst, u_tmp), dim=0)

                    ins_a = u_tmp
                    ins_a[..., 2:] = torch.tensor(u_h).float()
                    u_tmp = u_tmp.unsqueeze(0).float().to(device)
                    # print("u_tmp", u_tmp.shape)

                    # Updates
                    # pdb.set_trace()
                    # hidden_s = (
                    #     tmp_hidden_s[:, best_idx[0][0], :].unsqueeze(1).unsqueeze(0)
                    # )
                    # hidden_a = (
                    #     tmp_hidden_a[:, best_idx[0][0], :].unsqueeze(1).unsqueeze(0)
                    # )
                    hidden_s = tmp_hidden_s[:, best_idx, :].unsqueeze(1).unsqueeze(0)
                    hidden_a = tmp_hidden_a[:, best_idx, :].unsqueeze(1).unsqueeze(0)
                    hidden_s_lst = torch.cat((hidden_s_lst, hidden_s), dim=0)
                    hidden_a_lst = torch.cat((hidden_a_lst, hidden_a), dim=0)

                    # hidden_a = tmp_hidden_a[:, best_idx[0][0], :].unsqueeze(1)

                    # Get true observation from simulator
                    print(
                        "Compare: ",
                        "human: ",
                        u_h_d,
                        "pred act: ",
                        u_tmp,
                        "next step act",
                        ins_a,
                    )

                    obs, reward, done, info = env.step(u_tmp.squeeze(), CONST_DT)
                    obs = env.standardize(torch.tensor(obs), mean_s, std_s)
                    obs_t = obs.unsqueeze(0).unsqueeze(0).float().to(device)
                    # print("OBST", obs_t.shape, "u_tmp", u_tmp.shape)
                # print("obst", obs_t.shape, "u_tmp", u_tmp.shape)
                # obs_lst = torch.cat((obs_lst, obs_t), dim=0)

                u_lst = torch.cat((u_lst, u_tmp.squeeze(0)), dim=0)
                obs_lst = torch.cat((obs_lst, obs_t.squeeze(0)), dim=0)
                # print("##########", n_iter, u_lst.shape, obs_lst.shape)

                print("Loop: ", loops, "n_iter", n_iter, info)
                next_game_tick += CONST_DT
                loops += 1
                n_iter += 1

        # UPDATE DISPLAY
        if not env.done:
            env.redraw()
            ##if callback is not None:
            ##    callback(prev_obs, obs, action, rew, env_done, info)
            # CLOCK TICK
        clock.tick(FPS)
        if clock.get_fps() > 0:
            print("Reported dt: ", 1 / clock.get_fps())

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key == 27:
                    running = False
            elif event.type == pygame.QUIT:
                running = False
    if obs is not None:
        rendered = env.render(mode="rgb_array")

print("Done.")
