""" Recurrent model training """
import argparse
import copy
import pdb
import random
from functools import partial
from os import mkdir
from os.path import join, exists, dirname, abspath

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import yaml
import time
import sys

sys.path.insert(1, dirname(abspath(__file__)) + "/../envs")
from utils import load_cfg
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# window size used for simulator when collecting data
WINDOW_W = 1200
WINDOW_H = 600

sys.path.insert(1, dirname(abspath(__file__)) + "/..")
yaml_filepath = join(dirname(__file__), "../config/inference_params.yml")
parser = argparse.ArgumentParser("WorldVAE training & validation")
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
data_base = cfg["data_base"]
mode = cfg["mode"]
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
fig_border = cfg["fig_border"]
teacher_forcing_ratio = cfg["teacher_forcing"]
freq = cfg["freq"]
epochs = cfg["epochs"]
time_limit = cfg["time_limit"]
beta_min = cfg["beta_min"]
beta_max = cfg["beta_max"]

map_cfg = load_cfg(join("../config/maps", cfg["map_config"]))

# Initialize other parameters
b_step = 1
counter = 0
beta_interval = 1000
max_beta_int = beta_interval + 2000
min_beta_int = beta_interval
beta = beta_min
ratio = 1.0


def unstandardize(ins, mean, std):
    ua = ins * std + mean
    return ua


def standardize(ins, mean, std):
    s_ins = (ins - mean) / std
    if torch.any(torch.isinf(s_ins)):
        print("ERROR: Inf in actions detected!")
        std += 1e-4
    if torch.any(torch.isnan(ins)):
        print("ERROR: Nan in actions detected!")
    return s_ins


# def plot_mse_state(state, save_dir, N=3, label=None, epoch=None, train=None, std=None):
#     print("here!")

#     # pdb.set_trace()
#     seq_len = state.shape[1]
#     T = [(i + 1) * 1 / 30.0 for i in range(seq_len)]
#     t = T[H:]  # autoreg
#     p = T[:H]  # tf
#     na = state[..., H:, :]  # autoreg
#     a = state[..., :H, :]  # tf
#     plt.rcParams["axes.edgecolor"] = str(fig_border)
#     plt.rcParams["axes.linewidth"] = 1

#     colors = [
#         "#1f77b4",
#         "#ff7f0e",
#         "#9467bd",
#         "#8c564b",
#         "#e377c2",
#         "#7f7f7f",
#         "#bcbd22",
#         "#17becf",
#         "#eeefff",
#         "#FA8072",
#     ]

#     for b in range(0, N, 1):
#         fig = plt.figure(figsize=(8, 6), dpi=100)
#         # Traj plot
#         pdb.set_trace()
#         plt.plot(
#             state[b, :, 0],
#             -state[b, :, 1],
#             ".",
#             c=colors[b],
#             alpha=0.3,
#             label="inference",
#         )
#         if label is None:
#             plt.plot(
#                 state[b, 0, 0],
#                 -state[b, 0, 1],
#                 "yx",
#                 markersize=20,
#                 label="START",
#                 linewidth=20,
#             )
#         else:
#             plt.plot(
#                 label[b, 0, 0],
#                 -label[b, 0, 1],
#                 "yx",
#                 markersize=20,
#                 label="START",
#                 linewidth=20,
#             )
#             plt.plot(label[b, :, 0], -label[b, :, 1], "r-", label="True trajectory")
#     plt.show()
#     # add goal
#     goal_pos_x = map_cfg["GOAL"][0] * WINDOW_W
#     goal_pos_y = map_cfg["GOAL"][1] * WINDOW_H

#     plt.plot(goal_pos_x, -goal_pos_y, "g*", markersize=30, label="GOAL")

#     # add obstacles
#     ca = plt.gca()

#     # print(map_cfg["OBSTACLES"])
#     obstacle_cnt = map_cfg["OBSTACLES"][0]["COUNT"]
#     obstacle_pos = map_cfg["OBSTACLES"][0]["POSITIONS"]
#     obstacle_size = map_cfg["OBSTACLES"][0]["SIZES"]
#     # pygame pos is top left, matplotlib pos is bottom left
#     for ob in range(obstacle_cnt):
#         obstacle_w = obstacle_size[ob][0]
#         obstacle_h = obstacle_size[ob][1]
#         obstacle_x = obstacle_pos[ob][0] * WINDOW_W - obstacle_w / 2.0
#         obstacle_y = -obstacle_pos[ob][1] * WINDOW_H - obstacle_h / 2.0
#         ca.add_patch(
#             patches.Rectangle(
#                 (obstacle_x, obstacle_y),
#                 obstacle_w,
#                 obstacle_h,
#                 facecolor=(1.0, 0.0, 0.0, 0.9),
#                 zorder=2,
#             )
#         )

#         # obstacle_x = 0.5 * WINDOW_W + obstacle_w / 2.0
#         # obstacle_y = 0.5 * WINDOW_H + obstacle_h / 2.0
#         # ca.add_patch(patches.Rectangle((obstacle_x, -obstacle_y), obstacle_w, obstacle_h, facecolor=(1.0, 0.0, 0.0, 0.9), zorder=2))

#     # ca.add_patch(patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none'))

#     plt.rcParams["pdf.fonttype"] = 42
#     # Options
#     # params = {'text.usetex' : True,
#     #         'font.size' : 11,
#     #         'font.family' : 'lmodern',
#     #         'text.latex.unicode': True,
#     #         }
#     # plt.rcParams.update(params)
#     plt.gca().set_aspect("equal")  # , adjustable='box')
#     # plt.axis('square')
#     # plt.xlabel("x (pixels)")
#     # plt.ylabel("y (pixels)")
#     plt.xlim([0, WINDOW_W])
#     plt.ylim([0, -WINDOW_H])
#     # plt.axis('off')
#     plt.gca().axes.get_yaxis().set_visible(False)
#     plt.gca().axes.get_xaxis().set_visible(False)

#     fname = join(save_dir, "step_" + str(epoch) + "-" + map_cfg["NAME"] + "-traj.png")
#     # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#     #   fancybox=True, shadow=True, ncol=5, labelspacing=2)
#     plt.savefig(fname, format="png", bbox_inches="tight")
#     plt.close("all")


def plot_mse_state(state, save_dir, N=3, label=None, epoch=None, train=None, std=None):
    print("here!")

    seq_len = state.shape[1]
    T = [(i + 1) * 1 / 30.0 for i in range(seq_len)]
    t = T[H:]  # autoreg
    p = T[:H]  # tf
    na = state[..., H:, :]  # autoreg
    a = state[..., :H, :]  # tf
    plt.rcParams["axes.edgecolor"] = str(fig_border)
    plt.rcParams["axes.linewidth"] = 1

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#eeefff",
        "#FA8072",
    ]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    # Traj plot
    for n in range(N):
        # plt.plot(
        #     state[n, :H, 0],
        #     -state[n, :H, 1],
        #     "+",
        #     c=colors[n],
        #     alpha=0.3,
        #     label="observations",
        # )
        plt.plot(
            state[n, :, 0],
            -state[n, :, 1],
            ".",
            c=colors[n],
            alpha=0.3,
            label="inference",
        )
    if label is None:
        pass
    else:
        plt.plot(
            label[n, 0, 0],
            -label[n, 0, 1],
            "yx",
            markersize=20,
            label="START",
            linewidth=20,
        )
        plt.plot(label[n, :, 0], -label[n, :, 1], "r-", label="True trajectory")
    # add goal
    goal_pos_x = map_cfg["GOAL"][0] * WINDOW_W
    goal_pos_y = map_cfg["GOAL"][1] * WINDOW_H

    plt.plot(goal_pos_x, -goal_pos_y, "g*", markersize=30, label="GOAL")

    # add obstacles
    ca = plt.gca()

    # print(map_cfg["OBSTACLES"])
    obstacle_cnt = map_cfg["OBSTACLES"][0]["COUNT"]
    obstacle_pos = map_cfg["OBSTACLES"][0]["POSITIONS"]
    obstacle_size = map_cfg["OBSTACLES"][0]["SIZES"]
    # pygame pos is top left, matplotlib pos is bottom left
    for ob in range(obstacle_cnt):
        obstacle_w = obstacle_size[ob][0]
        obstacle_h = obstacle_size[ob][1]
        obstacle_x = obstacle_pos[ob][0] * WINDOW_W - obstacle_w / 2.0
        obstacle_y = -obstacle_pos[ob][1] * WINDOW_H - obstacle_h / 2.0
        ca.add_patch(
            patches.Rectangle(
                (obstacle_x, obstacle_y),
                obstacle_w,
                obstacle_h,
                facecolor=(1.0, 0.0, 0.0, 0.9),
                zorder=2,
            )
        )

        # obstacle_x = 0.5 * WINDOW_W + obstacle_w / 2.0
        # obstacle_y = 0.5 * WINDOW_H + obstacle_h / 2.0
        # ca.add_patch(patches.Rectangle((obstacle_x, -obstacle_y), obstacle_w, obstacle_h, facecolor=(1.0, 0.0, 0.0, 0.9), zorder=2))

    # ca.add_patch(patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none'))

    plt.rcParams["pdf.fonttype"] = 42
    # Options
    # params = {'text.usetex' : True,
    #         'font.size' : 11,
    #         'font.family' : 'lmodern',
    #         'text.latex.unicode': True,
    #         }
    # plt.rcParams.update(params)
    plt.gca().set_aspect("equal")  # , adjustable='box')
    # plt.axis('square')
    # plt.xlabel("x (pixels)")
    # plt.ylabel("y (pixels)")
    plt.xlim([0, WINDOW_W])
    plt.ylim([0, -WINDOW_H])
    # plt.axis('off')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    fname = join(save_dir, "step_" + str(epoch) + "-" + map_cfg["NAME"] + "-traj.png")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #   fancybox=True, shadow=True, ncol=5, labelspacing=2)
    plt.savefig(fname, format="png", bbox_inches="tight")
    plt.close("all")


def compute_fluency_cont(action) -> float:
    player_1 = action[:, :, :2]
    player_2 = action[:, :, 2:]
    fluency = {}
    fluency["inter_f"] = []
    fluency["h_idle"] = []
    fluency["r_idle"] = []
    fluency["conf"] = []
    fluency["f_del"] = []
    # Interactive forces dot prod of forces is negative if compress/stretch
    if np.dot(player_1, player_2) < 0:
        fluency["inter_f"].append(1)
    else:
        fluency["inter_f"].append(0)
    # Human Idle if all actions are 0, then it is idle
    if not np.any(player_2):
        fluency["h_idle"].append(1)
    else:
        fluency["h_idle"].append(0)
    # Robot Idle
    if not np.any(player_1):
        fluency["r_idle"].append(1)
    else:
        fluency["r_idle"].append(0)
    # Concurrent action: when both are acting
    if np.any(player_2) and np.any(player_1):
        fluency["conf"].append(1)
    else:
        fluency["conf"].append(0)
    # Funct. delay: when both are not acting
    if (not np.any(player_2)) and (not np.any(player_1)):
        fluency["f_del"].append(1)
    else:
        fluency["f_del"].append(0)
    return fluency


def plot_mse_act(act, label, epoch, train, std=None):
    folder = "viz_pred-"
    # A = True
    seq_len = act.shape[1]
    T = [(i + 1) * 1 / 30.0 for i in range(seq_len)]
    t = T[H:]  # autoreg
    p = T[:H]  # tf
    na = act[:, H:, ...]  # autoreg
    a = act[:, :H, ...]  # tf
    # if A:
    #    T = p
    for b in range(0, BSIZE, b_step):

        fig, axs = plt.subplots(4, 1, figsize=(15, 7))
        axs[0].plot(p, act[b, :H, 0], "g-", label="pred, tf")
        axs[0].plot(t, act[b, H:, 0], "b-", label="pred, ag")
        axs[0].plot(T, label[b, :, 0], "r-", label="true")
        axs[0].fill_between(
            p,
            act[b, :H, 0] - 2 * (std[b, :H, 0]),
            act[b, :H, 0] + 2 * (std[b, :H, 0]),
            color="b",
            alpha=0.5,
        )
        axs[0].fill_between(
            t,
            act[b, H:, 0] - 2 * (std[b, H:, 0]),
            act[b, H:, 0] + 2 * (std[b, H:, 0]),
            color="b",
            alpha=0.5,
        )
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Force Units")

        axs[1].plot(p, act[b, :H, 1], "g-", label="pred, tf")
        axs[1].plot(t, act[b, H:, 1], "b-", label="pred, ag")
        axs[1].plot(T, label[b, :, 1], "r-", label="true")
        axs[1].fill_between(
            p,
            act[b, :H, 1] - 2 * (std[b, :H, 1]),
            act[b, :H, 1] + 2 * (std[b, :H, 1]),
            color="b",
            alpha=0.5,
        )
        axs[1].fill_between(
            t,
            act[b, H:, 1] - 2 * (std[b, H:, 1]),
            act[b, H:, 1] + 2 * (std[b, H:, 1]),
            color="b",
            alpha=0.5,
        )
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Force Units")

        axs[2].plot(p, act[b, :H, 2], "g-", label="pred, tf")
        axs[2].plot(t, act[b, H:, 2], "b-", label="pred, ag")
        axs[2].plot(T, label[b, :, 2], "r-", label="true")
        axs[2].fill_between(
            p,
            act[b, :H, 2] - 2 * (std[b, :H, 2]),
            act[b, :H, 2] + 2 * (std[b, :H, 2]),
            color="b",
            alpha=0.5,
        )
        axs[2].fill_between(
            t,
            act[b, H:, 2] - 2 * (std[b, H:, 2]),
            act[b, H:, 2] + 2 * (std[b, H:, 2]),
            color="b",
            alpha=0.5,
        )
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Force Units")

        axs[3].plot(p, act[b, :H, 3], "g-", label="pred, tf")
        axs[3].plot(t, act[b, H:, 3], "b-", label="pred, ag")
        axs[3].plot(T, label[b, :, 3], "r-", label="true")
        axs[3].fill_between(
            p,
            act[b, :H, 3] - 2 * (std[b, :H, 3]),
            act[b, :H, 3] + 2 * (std[b, :H, 3]),
            color="b",
            alpha=0.5,
        )
        axs[3].fill_between(
            t,
            act[b, H:, 3] - 2 * (std[b, H:, 3]),
            act[b, H:, 3] + 2 * (std[b, H:, 3]),
            color="b",
            alpha=0.5,
        )
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Force Units")
        for i in range(4):
            axs[i].set_ylim(-2, 2)

        fname = join(
            sample_dir,
            folder
            + str(b)
            + "-"
            + map_cfg["NAME"]
            + "_count-"
            + str(epoch)
            + "actions.png",
        )
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=5,
            labelspacing=2,
        )

        plt.savefig(fname, format="png")
        plt.close("all")

        # # Plots for MSE
        # tx1, ty1 = h1t[:, 0], h1t[:, 1]  # True x, y for h1
        # tx2, ty2 = h2t[:, 0], h2t[:, 1]  # True x, y for h2
        # x1, y1 = h1[:, 0], h1[:, 1]  # Pred x, y for all h1
        # x2, y2 = h2[:, 0], h2[:, 1]  # Pred x, y for all h2
        # nh1 = np.asarray([x1, y1])
        # nh2 = np.asarray([x2, y2])
        # th1 = np.asarray([tx1, ty1])
        # th2 = np.asarray([tx2, ty2])
        # mse_1 = np.linalg.norm(nh1 - th1, axis=0)
        # mse_2 = np.linalg.norm(nh2 - th2, axis=0)
        # fig = plt.figure()
        # plt.plot(mse_1, "r", label="h1 error")
        # plt.plot(mse_2, "b", label="h2 error")
        # plt.xlabel("time")
        # plt.ylabel("mse")
        # fname = join(
        #     sample_dir, folder + str(b) + "/count-" + str(epoch) + "mse_act.png"
        # )
        # fig.legend()
        # plt.savefig(fname, format="png")
        # plt.close("all")


def get_loss(
    state, action, reward, next_state, next_action, epoch, time_limit, ratio, train
):
    """Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """

    #    print(action[1], action[2], next_action[1], '\n\n\n')
    if control_type == "discrete":
        action_scaled = action.type(torch.LongTensor).to(
            device, non_blocking=True
        )  # standardize(action, mean_a, std_a)
        next_action_scaled = (
            next_action.type(torch.LongTensor)
            .to(device, non_blocking=True)
            .reshape(H * BSIZE, -1)
            .squeeze()
        )  # standardize(next_action, mean_a, std_a)
    else:
        # action_scaled = standardize(action, mean_a, std_a).to(device, non_blocking=True)
        # next_action_scaled = standardize(next_action, mean_na, std_na).to(
        #     device, non_blocking=True
        # )
        action_scaled = action.to(device, non_blocking=True)
        next_action_scaled = next_action.to(device, non_blocking=True)
    # reward_scaled = standardize(reward, mean_r, std_r)
    state_scaled = standardize(state, mean_s, std_s).float()
    next_state_scaled = standardize(next_state, mean_ns, std_ns)
    # state_scaled = state.to(device, non_blocking=True)
    # next_state_scaled = next_state.to(device, non_blocking=True)

    # for plots
    # if epoch % 10 == 0:
    N = NGAUSS
    pred_act = torch.zeros((BSIZE, SEQ_LEN, ASIZE)).to(device, non_blocking=True)
    sig_a_lst = torch.zeros((BSIZE, SEQ_LEN, ASIZE)).to(device, non_blocking=True)
    pred_s = torch.zeros((BSIZE, N, SEQ_LEN, LSIZE)).to(device, non_blocking=True)
    sig_s_lst = torch.zeros((BSIZE, N, SEQ_LEN, LSIZE)).to(device, non_blocking=True)

    if control_type == "discrete":
        logits, probs, hiddens = worldvae(action_scaled, state_scaled, H)
    else:
        for n in range(N):
            (
                mus,
                sigmas,
                logpis,
                mu_a,
                sig_a,
                logpi_a,
                hidden_tf,
                hidden_tf_a,
            ) = worldvae(
                action_scaled[:, :H, ...], state_scaled[:, :H, ...], h=None, h_a=None
            )
            # mus, sigmas, logpis, mu_a, sig_a, logpi_a, hidden_tf = worldvae(
            #         action_scaled[:, :H, ...], state_scaled[:, :H, ...], h=None
            # )
            # print('init hidden:', hidden_tf.shape, 'action_scaled', action_scaled.shape, 'state_scaled', state_scaled.shape)
            batch_a = next_action_scaled[:, :H, ...]
            act_h_tf = gmm_loss(batch_a, mu_a, sig_a, logpi_a)
            if device == torch.device("cpu"):
                na, j = worldvae.get_gmm_sample(
                    mu_a.detach().clone(),
                    sig_a.detach().clone(),
                    torch.exp(logpi_a.detach().clone()),
                    False,
                )
            else:
                na, j = worldvae.module.get_gmm_sample(
                    mu_a.detach().clone(),
                    sig_a.detach().clone(),
                    torch.exp(logpi_a.detach().clone()),
                    False,
                )
            loss_tf_a = act_h_tf / ASIZE

            batch_s = next_state_scaled[:, :H, ...]
            gmm = gmm_loss(batch_s, mus, sigmas, logpis)
            if device == torch.device("cpu"):
                # ns, k = worldvae.get_gmm_sample(
                #     mus.detach().clone(),
                #     sigmas.detach().clone(),
                #     torch.exp(logpis.detach().clone()),
                #     False,
                # )
                ns, k = worldvae.get_gmm_sample_from_mode(
                    mus.detach().clone(),
                    sigmas.detach().clone(),
                    n,
                    False,
                )
            else:
                ns, k = worldvae.module.get_gmm_sample(
                    mus.detach().clone(),
                    sigmas.detach().clone(),
                    torch.exp(logpis.detach().clone()),
                    False,
                )
            loss_tf_s = gmm / LSIZE

            sig = sig_a[:, :, j, :].squeeze()  # .squeeze(1) if H=1

            sig_a_lst[:, :H, ...] = sig[:, :H, ...]
            pred_act[:, :H, ...] = na

            sig = sigmas[:, :, k, :].squeeze()  # .squeeze(1)

            sig_s_lst[:, n, :H, ...] = sig[:, :H, ...]
            pred_s[:, n, :H, ...] = ns

            # print('na', na.shape, next_action_scaled.shape)
            mse_a_tf = f.mse_loss(batch_a, na)
            mse_s_tf = f.mse_loss(
                next_state_scaled[:, :H, ...], ns
            )  # unstandardize(ns, mean_ns, std_ns))

            loss_ntf_a = 0.0
            loss_ntf_s = 0.0
            mse_a_ntf = 0.0
            mse_s_ntf = 0.0
            hidden = hidden_tf
            hidden_a = hidden_tf_a
            # print('as', action_scaled.shape)
            ins_a = action_scaled[:, H, ...].unsqueeze(1)
            ins_s = state_scaled[:, H, ...].unsqueeze(1)
            for t in range(H, SEQ_LEN):
                # print('ENTERED')
                # print('t', t, 'hidden', hidden.shape, 'hidden_a', hidden_a.shape,'ins_s', ins_s.shape, 'ins_a', ins_a.shape)
                mus, sigmas, logpis, mu_a, sig_a, logpi_a, hidden, hidden_a = worldvae(
                    actions=ins_a, states=ins_s, h=hidden, h_a=hidden_a
                )
                # mus, sigmas, logpis, mu_a, sig_a, logpi_a, hidden = worldvae(
                #         actions=ins_a, states=ins_s, h=hidden
                # )
                batch_a = next_action_scaled[:, t, ...].unsqueeze(1)
                act_h_ntf = gmm_loss(batch_a, mu_a, sig_a, logpi_a)
                loss_ntf_a += act_h_ntf / ASIZE
                if device == torch.device("cpu"):
                    na, j = worldvae.get_gmm_sample(
                        mu_a.detach().clone(),
                        sig_a.detach().clone(),
                        torch.exp(logpi_a.detach().clone()),
                        False,
                    )
                else:
                    na, j = worldvae.module.get_gmm_sample(
                        mu_a.detach().clone(),
                        sig_a.detach().clone(),
                        torch.exp(logpi_a.detach().clone()),
                        False,
                    )
                sig = sig_a[:, :, j, :]
                sig_a_lst[:, t, :] = sig.squeeze()
                pred_act[:, t, ...] = na.squeeze()
                ins_a = na

                batch_s = next_state_scaled[:, t, ...].unsqueeze(1)
                gmm = gmm_loss(batch_s, mus, sigmas, logpis)
                loss_ntf_s += gmm / LSIZE
                if device == torch.device("cpu"):
                    # ns, k = worldvae.get_gmm_sample(
                    #     mus.detach().clone(),
                    #     sigmas.detach().clone(),
                    #     torch.exp(logpis.detach().clone()),
                    #     False,
                    # )
                    ns, k = worldvae.get_gmm_sample_from_mode(
                        mus.detach().clone(),
                        sigmas.detach().clone(),
                        n,
                        False,
                    )
                else:
                    ns, k = worldvae.module.get_gmm_sample(
                        mus.detach().clone(),
                        sigmas.detach().clone(),
                        torch.exp(logpis.detach().clone()),
                        False,
                    )
                sig = sigmas[:, :, k, :]
                sig_s_lst[:, n, t, :] = sig.squeeze()
                pred_s[:, n, t, ...] = ns.squeeze(1)
                ins_s = ns

                mse_a_ntf += f.mse_loss(batch_a, na)
                mse_s_ntf += f.mse_loss(
                    next_state_scaled[:, t, ...].unsqueeze(1), ns
                )  # , unstandardize(ns, mean_ns, std_ns))

    # Losses
    loss = (
        (1 - ratio) * (loss_tf_a + loss_ntf_a + loss_tf_s + loss_ntf_s) / N
    )  # + ratio * (mse_s_tf + mse_s_ntf + mse_a_tf + mse_a_ntf)
    # batch = next_state_scaled
    mse_r = 0
    # gmm = gmm_loss(batch, mus, sigmas, logpi)
    gmm = 0
    # mse_r = f.mse_loss(rs.squeeze(), reward_scaled.squeeze())
    # print("R pred ", rs)
    # print("R true scaled ", reward_scaled)
    # act_h = 0
    # act_h = f.cross_entropy(logits, next_action_scaled.type(torch.LongTensor).to(device, non_blocking=True))
    #    bce = f.binary_cross_entropy_with_logits(ds[:last_idx, ...], terminal[:last_idx, ...])
    # loss = (gmm + mse_r + act_h) / (LSIZE + 1 + 1)
    # pdb.set_trace()
    plot_mse_act(
        pred_act.contiguous().squeeze().cpu().numpy(),
        next_action.squeeze().contiguous().cpu().numpy(),
        epoch,
        train,
        std=sig_a_lst.detach().clone().cpu().numpy(),
    )
    pred_s_unstd = unstandardize(pred_s, mean_ns, std_ns).to(device, non_blocking=True)
    plot_mse_state(
        pred_s_unstd.contiguous().cpu().numpy(),
        next_state.contiguous().cpu().numpy(),
        epoch,
        train,
        std=sig_s_lst.detach().clone().cpu().numpy(),
    )

    return (
        dict(
            loss=loss,
            loss_tf_a=loss_tf_a,
            loss_ntf_a=loss_ntf_a,
            loss_tf_s=loss_tf_s,
            loss_ntf_s=loss_ntf_s,
            mse_s_tf=mse_s_tf,
            mse_s_ntf=mse_s_ntf,
            mse_a_tf=mse_a_tf,
            mse_a_ntf=mse_a_ntf,
            gmm=gmm,
            mse_r=mse_r,
        ),
        ratio,
    )
