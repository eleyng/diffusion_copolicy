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
from plot import plot_mse_state
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
teacher_forcing_ratio = cfg["teacher_forcing"]
freq = cfg["freq"]
epochs = cfg["epochs"]
time_limit = cfg["time_limit"]
beta_min = cfg["beta_min"]
beta_max = cfg["beta_max"]


# Initialize other parameters
ratio = 0.0
K = NGAUSS

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
# mean_r = torch.Tensor(train_stats["mean_r"]).to(device, non_blocking=True)
# std_r = torch.Tensor(train_stats["std_r"]).to(device, non_blocking=True)
# mean_a = torch.Tensor(train_stats["mean_a"]).to(device, non_blocking=True)
# std_a = torch.Tensor(train_stats["std_a"]).to(device, non_blocking=True)
# mean_na = torch.Tensor(train_stats["mean_na"]).to(device, non_blocking=True)
# std_na = torch.Tensor(train_stats["std_na"]).to(device, non_blocking=True)


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

u = []  # .to(device, non_blocking=True)

sig_a_lst = torch.zeros((BSIZE, SEQ_LEN, ASIZE)).to(device, non_blocking=True)
pred_act = torch.zeros((BSIZE, SEQ_LEN, ASIZE)).to(device, non_blocking=True)
hidden_a = torch.zeros((n_layers, 1, RSIZE)).to(device, non_blocking=True)

sig_s_lst = torch.zeros((BSIZE, SEQ_LEN, LSIZE)).to(device, non_blocking=True)
pred_s = torch.zeros((BSIZE, SEQ_LEN, LSIZE)).to(device, non_blocking=True)
hidden_s = torch.zeros((n_layers, 1, RSIZE)).to(device, non_blocking=True)


initial_observations = []  # .to(device, non_blocking=True)
initial_actions = []  # .to(device, non_blocking=True)
rollout_observations = []  # .to(device, non_blocking=True)
rollout_actions = []  # .to(device, non_blocking=True)

while running and n_iter < n_steps:
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
                u_h[0] = u_h_d[0]
                u_h[1] = u_h_d[1]
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
                    u_tmp[:, :, 2:] = u_h
                    u_tmp = torch.tensor(u_tmp).float().to(device)
                    # u_t = u_tmp.squeeze(0)  # (1,1,4) --> (1,4)

                    # Get true observation from simulator
                    obs, reward, done, info = env.step(u_tmp.squeeze(), CONST_DT)
                    obs = env.standardize(torch.tensor(obs), mean_s, std_s)

                    # Create obs tensor
                    obs_t = obs.unsqueeze(0).unsqueeze(0).float().to(device)

                    # Update hidden state only
                    # _, _, _, _, _, _, hidden_s, hidden_a = model(
                    #     actions=u_tmp, states=obs_t, h=hidden_s, h_a=hidden_a
                    # )
                    _, _, _, hidden_s = model(actions=u_tmp, states=obs_t, h=hidden_s)
                    u.append(u_tmp.squeeze(0))

                # n_iter >= H: robot plan & act
                else:
                    # pdb.set_trace()
                    print("*****PREDICTING*****")
                    # Rollout  futures up to length SEQ_LEN
                    pred_act = torch.zeros((K, SEQ_LEN, ASIZE)).to(
                        device, non_blocking=True
                    )
                    sig_a_lst = torch.zeros((K, SEQ_LEN, ASIZE)).to(
                        device, non_blocking=True
                    )
                    pred_s = torch.zeros((K, SEQ_LEN, LSIZE)).to(
                        device, non_blocking=True
                    )
                    sig_s_lst = torch.zeros((K, SEQ_LEN, LSIZE)).to(
                        device, non_blocking=True
                    )
                    mu_lst = torch.zeros((K, SEQ_LEN, NGAUSS, LSIZE + ASIZE)).to(
                        device, non_blocking=True
                    )
                    sig_lst = torch.zeros((K, SEQ_LEN, NGAUSS, LSIZE + ASIZE)).to(
                        device, non_blocking=True
                    )

                    ins_a = u_tmp.repeat(K, 1, 1)
                    ins_s = obs_t.repeat(K, 1, 1)
                    # action = torch.tensor(u_t).repeat(K, 1, 1).to(device)  # B x T x F

                    # obs_tmp = obs.repeat(K, 1).unsqueeze(1).to(device)  # B x T x F

                    # hidden_a_tmp = hidden_a.repeat(1, K, 1).to(
                    #     device
                    # )  # num_layer x B x F
                    # print("HIDDEN A", hidden_a_tmp.size(), hidden_a.size())
                    # hidden_s_tmp = hidden_s.repeat(1, K, 1).to(
                    #     device
                    # )  # num_layer x B x F
                    hidden = hidden_s.repeat(1, K, 1)

                    cum_r = np.zeros((K, 1))  # B x T
                    for t in range(SEQ_LEN):

                        (
                            mus,
                            sigmas,
                            logpis,
                            # mu_a,
                            # sig_a,
                            # logpi_a,
                            hidden,
                            # hidden_a,
                        ) = model(
                            actions=ins_a,
                            states=ins_s.float(),
                            h=hidden,
                            # h_a=hidden_a_tmp,
                        )

                        # mu_a = mus[..., LSIZE:]
                        # sig_a = sigmas[..., LSIZE:]

                        # mu_s = mus[..., :LSIZE]
                        # sig_s = sigmas[..., :LSIZE]
                        # mus, sigmas, logpis, mu_a, sig_a, logpi_a, _, hidden = model(
                        #        actions=ins_a, states=ins_s, h=hidden
                        # )

                        # na, j = model.get_gmm_sample(
                        #     mu_a.detach().clone(),
                        #     sig_a.detach().clone(),
                        #     torch.exp(logpi_a.detach().clone()),
                        #     False,
                        # )
                        # nsa = model.get_gmm_sample_from_mode(
                        #     mus.detach().clone(),
                        #     sigmas.detach().clone(),
                        #     trick=False,
                        # )
                        # ns = nsa[..., :LSIZE]
                        # na = nsa[..., LSIZE:]

                        mu_lst[:, t, :, :] = mus.squeeze()
                        sig_lst[:, t, :, :] = sigmas.squeeze()

                        # sig = sigmas[:, :, j, :]
                        # sig_a_lst[:, t, :] = sig.squeeze()
                        # pred_act[:, t, ...] = na.squeeze()

                        # ns, k = model.get_gmm_sample(
                        #     mus.detach().clone(),
                        #     sigmas.detach().clone(),
                        #     torch.exp(logpis.detach().clone()),
                        #     False,
                        # )

                        # sig = sigmas[:, :, k, :]
                        # sig_s_lst[:, t, :] = sig.squeeze()
                        # pred_s[:, t, ...] = ns.squeeze(1)

                        # sig = sig_a[:, :, k, :].squeeze()
                        # sig_a_lst[:, :H, ...] = sig[:, :H, ...]
                        # pred_act[:, :H, ...] = na

                        # sig = sig_s[:, :, k, :].squeeze()
                        # sig_s_lst[:, :H, ...] = sig[:, :H, ...]
                        # pred_s[:, :H, ...] = ns.squeeze(1)

                        if t == 0:
                            tmp_hidden = hidden
                            # tmp_hidden_a = hidden_a_tmp

                        for k in range(K):
                            nsa, _ = model.get_gmm_sample_from_mode(
                                mus.detach().clone(),
                                sigmas.detach().clone(),
                                idx=k,
                                trick=False,
                            )

                            pred_act[k, t, ...] = nsa.squeeze()[k, LSIZE:]
                            pred_s[k, t, ...] = nsa.squeeze()[k, :LSIZE]

                        ins_a = pred_act[:, t, :].unsqueeze(1)
                        ins_s = pred_s[:, t, :].unsqueeze(1)

                    # Compute reward using state
                    rs = np.zeros((K,))
                    for k in range(K):
                        rs_k = 0
                        for t in range(SEQ_LEN):
                            # spdb.set_trace()
                            rs_k = env_r.compute_reward(
                                state=pred_s[k, t, :3].detach().numpy()
                            )
                            # pdb.set_trace()
                        rs[k] = rs_k
                    # pdb.set_trace()

                    cum_r = rs  # np.add(cum_r, rs)

                    pred_s[..., 1] = -pred_s[..., 1]
                    pred_s_unstd = env.unstandardize(pred_s, mean_ns, std_ns).to(
                        device, non_blocking=True
                    )
                    # pdb.set_trace()
                    if n_iter % 30 == 0:
                        plot_mse_state(
                            pred_s_unstd.contiguous().cpu().numpy(),
                            save_dir=plt_pred_dir,
                            N=K,
                            epoch=env.n_step,
                            # std=sig_s_lst.detach().clone().cpu().numpy(),
                        )

                    # Select best action
                    # print("Reward", cum_r)
                    best_idx = np.where(cum_r.squeeze() == cum_r.max())
                    # print("Best idx", best_idx[0][0], u_tmp.shape)
                    u_tmp = pred_act[best_idx[0][0], 0, :]  # .detach().numpy()
                    u_tmp = torch.tensor(u_tmp).float().to(device)
                    u.append(u_tmp)

                    ins_a[..., :2] = u_tmp[..., :2]
                    ins_a[..., 2:] = torch.tensor(u_h).float()

                    # Updates
                    hidden = tmp_hidden  # [:, best_idx[0][0], :].unsqueeze(1)
                    # hidden_a = tmp_hidden_a[:, best_idx[0][0], :].unsqueeze(1)

                    # Get true observation from simulator
                    print(
                        "Compare: ",
                        "human: ",
                        u_h_d,
                        "chosen act: ",
                        u_tmp,
                        "next step act",
                        ins_a[0, ...],
                    )
                    obs, reward, done, info = env.step(u_tmp.squeeze(), CONST_DT)
                    obs = env.standardize(torch.tensor(obs), mean_s, std_s)
                    ins_s = obs.unsqueeze(0).unsqueeze(0).float().to(device)

                print("Loop: ", loops, "n_iter", n_iter, info)
                next_game_tick += CONST_DT
                loops += 1
                n_iter += 1

            if obs is not None:
                rendered = env.render(mode="rgb_array")

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

print("Done.")
