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
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


from cooperative_transport.gym_table.envs.utils import load_cfg

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


# window size used for simulator when collecting data
WINDOW_W = 1200
WINDOW_H = 600

yaml_filepath = join("cooperative_transport/gym_table/config", "inference_params.yml")
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
mode = cfg["mode"]
experiment_name = cfg["experiment_name"]
exp_string = cfg["exp_string"]
print(experiment_name)
train_stats_f = cfg["train_stats_f"]
# data_dir = cfg["data_dir"]
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

map_cfg = load_cfg(
    join("cooperative_transport/gym_table/config/maps", cfg["map_config"])
)

# Initialize other parameters
b_step = 1
counter = 0
beta_interval = 1000
max_beta_int = beta_interval + 2000
min_beta_int = beta_interval
beta = beta_min
ratio = 1.0


def plot_mse_state(state, epoch, save_dir=None, label=None, std=None, start=None):
    # print("here!", state.shape, label.shape, std.shape)
    folder = "valid/NAgauss-"

    seq_len = state.shape[1]

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
        "#ffbb11",
        "#2ca02c",
        "#d62728",
        "#bbf90f",
        "#00FFFF",
        "#000080",
        "#FFFF00",
        "#008080",
    ]

    for b in range(0, BSIZE, b_step):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        # Traj plot
        for n in range(NGAUSS * AGAUSS):
            plt.plot(
                state[b, :H, n, 0], -state[b, :H, n, 1], "+", c=colors[n], alpha=0.3
            )
            plt.plot(
                state[b, H:, n, 0], -state[b, H:, n, 1], ".", c=colors[n], alpha=0.3
            )
            if label is not None:
                plt.plot(
                    label[b, 0, 0],
                    -label[b, 0, 1],
                    "yx",
                    markersize=20,
                    label="START",
                    linewidth=20,
                )
                plt.plot(label[b, :, 0], -label[b, :, 1], "r-", label="True trajectory")
            else:
                plt.plot(
                    start[..., 0],  # state[b, H, n, 0],
                    -start[..., 1],  # -state[b, H, n, 0],
                    "yx",
                    markersize=20,
                    label="START",
                    linewidth=20,
                )
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

            # Options
            plt.gca().set_aspect("equal")  # , adjustable='box')
            # plt.axis('square')
            # plt.xlabel("x (pixels)")
            # plt.ylabel("y (pixels)")
            plt.xlim([0, WINDOW_W])
            plt.ylim([0, -WINDOW_H])
            # plt.axis('off')
            # plt.gca().axes.get_yaxis().set_visible(False)
            # plt.gca().axes.get_xaxis().set_visible(False)

            plt.rcParams["pdf.fonttype"] = 42

            plt.xlim([0, WINDOW_W])
            plt.ylim([0, -WINDOW_H])
            # plt.axis('off')

            fname = join(
                save_dir,
                "n_step-"
                + str(epoch)
                + "-NA-"
                + str(n)
                + exp_string
                + map_cfg["NAME"]
                + "-traj.png",
            )
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
    folder = "valid/NA-"
    # A = True
    seq_len = act.shape[1]
    T = [(i + 1) * 1 / 30.0 for i in range(seq_len)]
    t = T[H:]  # autoreg
    p = T[:H]  # tf
    na = act[:, H:, ...]  # autoreg
    a = act[:, :H, ...]  # tf
    # if A:
    #    T = p
    for n in range(NGAUSS * AGAUSS):
        for b in range(0, BSIZE, b_step):

            fig, axs = plt.subplots(4, 1, figsize=(15, 7))
            axs[0].plot(p, act[b, :H, n, 0], "g-", label="pred, tf")
            axs[0].plot(t, act[b, H:, n, 0], "b-", label="pred, ag")
            axs[0].plot(T, label[b, :, 0], "r-", label="true")
            axs[0].fill_between(
                p,
                act[b, :H, n, 0] - 2 * (std[b, :H, n, 0]),
                act[b, :H, n, 0] + 2 * (std[b, :H, n, 0]),
                color="b",
                alpha=0.5,
            )
            axs[0].fill_between(
                t,
                act[b, H:, n, 0] - 2 * (std[b, H:, n, 0]),
                act[b, H:, n, 0] + 2 * (std[b, H:, n, 0]),
                color="b",
                alpha=0.5,
            )
            axs[0].set_xlabel("Time [s]")
            axs[0].set_ylabel("Force Units")

            axs[1].plot(p, act[b, :H, n, 1], "g-", label="pred, tf")
            axs[1].plot(t, act[b, H:, n, 1], "b-", label="pred, ag")
            axs[1].plot(T, label[b, :, 1], "r-", label="true")
            axs[1].fill_between(
                p,
                act[b, :H, n, 1] - 2 * (std[b, :H, n, 1]),
                act[b, :H, n, 1] + 2 * (std[b, :H, n, 1]),
                color="b",
                alpha=0.5,
            )
            axs[1].fill_between(
                t,
                act[b, H:, n, 1] - 2 * (std[b, H:, n, 1]),
                act[b, H:, n, 1] + 2 * (std[b, H:, n, 1]),
                color="b",
                alpha=0.5,
            )
            axs[1].set_xlabel("Time [s]")
            axs[1].set_ylabel("Force Units")

            axs[2].plot(p, act[b, :H, n, 2], "g-", label="pred, tf")
            axs[2].plot(t, act[b, H:, n, 2], "b-", label="pred, ag")
            axs[2].plot(T, label[b, :, 2], "r-", label="true")
            axs[2].fill_between(
                p,
                act[b, :H, n, 2] - 2 * (std[b, :H, n, 2]),
                act[b, :H, n, 2] + 2 * (std[b, :H, n, 2]),
                color="b",
                alpha=0.5,
            )
            axs[2].fill_between(
                t,
                act[b, H:, n, 2] - 2 * (std[b, H:, n, 2]),
                act[b, H:, n, 2] + 2 * (std[b, H:, n, 2]),
                color="b",
                alpha=0.5,
            )
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("Force Units")

            axs[3].plot(p, act[b, :H, n, 3], "g-", label="pred, tf")
            axs[3].plot(t, act[b, H:, n, 3], "b-", label="pred, ag")
            axs[3].plot(T, label[b, :, 3], "r-", label="true")
            axs[3].fill_between(
                p,
                act[b, :H, n, 3] - 2 * (std[b, :H, n, 3]),
                act[b, :H, n, 3] + 2 * (std[b, :H, n, 3]),
                color="b",
                alpha=0.5,
            )
            axs[3].fill_between(
                t,
                act[b, H:, n, 3] - 2 * (std[b, H:, n, 3]),
                act[b, H:, n, 3] + 2 * (std[b, H:, n, 3]),
                color="b",
                alpha=0.5,
            )
            axs[3].set_xlabel("Time [s]")
            axs[3].set_ylabel("Force Units")
            for i in range(4):
                axs[i].set_ylim(-2, 2)

            fname = join(
                sample_dir,
                folder + str(n) + "-" + map_cfg["NAME"] + exp_string + "actions.pdf",
            )
            plt.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=5,
                labelspacing=2,
            )

            plt.savefig(fname, format="pdf")
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

    if control_type == "discrete":
        action_scaled = action.type(torch.LongTensor).to(device, non_blocking=True)
        next_action_scaled = (
            next_action.type(torch.LongTensor)
            .to(device, non_blocking=True)
            .reshape(H * BSIZE, -1)
            .squeeze()
        )
    else:

        action_scaled = action.to(device, non_blocking=True)
        next_action_scaled = next_action.to(device, non_blocking=True)

    state_scaled = standardize(state, mean_s, std_s).float()
    next_state_scaled = standardize(next_state, mean_ns, std_ns)

    pred_act = torch.zeros((BSIZE, SEQ_LEN, NGAUSS * AGAUSS, ASIZE)).to(
        device, non_blocking=True
    )
    sig_a_lst = torch.zeros((BSIZE, SEQ_LEN, NGAUSS * AGAUSS, ASIZE)).to(
        device, non_blocking=True
    )
    pred_s = torch.zeros((BSIZE, SEQ_LEN, NGAUSS * AGAUSS, LSIZE)).to(
        device, non_blocking=True
    )
    sig_s_lst = torch.zeros((BSIZE, SEQ_LEN, NGAUSS * AGAUSS, LSIZE)).to(
        device, non_blocking=True
    )

    # action_scaled[:, :H, ...] = 0.0 * torch.ones(action_scaled[:, :H, ...].size()) # torch.randn(action_scaled[:, :H, ...].size()) #* torch.ones(action_scaled[:, :H, ...].size())
    action_scaled[:, :H, 0] = 1.0 * torch.ones(action_scaled[:, :H, 0].size())
    next_action_scaled[:, : H - 1, 0] = 1.0 * torch.ones(
        action_scaled[:, : H - 1, 0].size()
    )
    action_scaled[:, :H, 1] = 0.0 * torch.ones(action_scaled[:, :H, 0].size())
    next_action_scaled[:, : H - 1, 1] = 0.0 * torch.ones(
        action_scaled[:, : H - 1, 0].size()
    )
    pdb.set_trace()

    mus, sigmas, logpis, mu_a, sig_a, logpi_a, hidden_tf, hidden_tf_a = worldvae(
        action_scaled[:, :H, ...], state_scaled[:, :H, ...], h=None, h_a=None
    )

    batch_a = next_action_scaled[:, :H, ...]
    act_h_tf = gmm_loss(batch_a, mu_a, sig_a, logpi_a)
    loss_tf_a = act_h_tf / ASIZE

    batch_s = next_state_scaled[:, :H, ...]
    gmm = gmm_loss(batch_s, mus, sigmas, logpis)
    loss_tf_s = gmm / LSIZE

    for n in range(NGAUSS):
        ns, _ = worldvae.get_gmm_sample_from_mode(
            mus.detach().clone(),
            sigmas.detach().clone(),
            n,
            False,
        )

        for m in range(AGAUSS):
            sig = sigmas[:, :, n, :]
            # pdb.set_trace()
            sig_s_lst[:, :H, (NGAUSS * m) + n, :] = sig
            pred_s[:, :H, (NGAUSS * m) + n, :] = ns

            na, _ = worldvae.get_gmm_sample_from_mode(
                mu_a.detach().clone(),
                sig_a.detach().clone(),
                m,
                False,
            )
            sig = sig_a[:, :, m, :].squeeze()
            sig_a_lst[:, :H, (NGAUSS * m) + n, :] = sig
            pred_act[:, :H, (NGAUSS * m) + n, :] = na

    print("na", na.shape, next_action_scaled.shape)
    mse_a_tf = f.mse_loss(batch_a, na)
    mse_s_tf = f.mse_loss(
        next_state_scaled[:, :H, ...], ns
    )  # unstandardize(ns, mean_ns, std_ns))

    loss_ntf_a = 0.0
    loss_ntf_s = 0.0
    mse_a_ntf = 0.0
    mse_s_ntf = 0.0
    hidden = hidden_tf.repeat(1, NGAUSS * AGAUSS, 1)
    hidden_a = hidden_tf_a.repeat(1, NGAUSS * AGAUSS, 1)
    print("as", action_scaled.shape)
    ins_a = action_scaled[:, H, ...].unsqueeze(1).repeat(NGAUSS * AGAUSS, 1, 1)
    ins_s = state_scaled[:, H, ...].unsqueeze(1).repeat(NGAUSS * AGAUSS, 1, 1)
    print("\nins", ins_s.shape)

    for t in range(H, SEQ_LEN):
        print("\n", hidden_a.shape, hidden.shape, ins_a.shape, ins_s.shape)
        # print('ENTERED')
        # print('t', t, 'hidden', hidden.shape, 'hidden_a', hidden_a.shape,'ins_s', ins_s.shape, 'ins_a', ins_a.shape)
        mus, sigmas, logpis, mu_a, sig_a, logpi_a, hidden, hidden_a = worldvae(
            actions=ins_a, states=ins_s, h=hidden, h_a=hidden_a
        )
        # mus, sigmas, logpis, mu_a, sig_a, logpi_a, _, hidden = worldvae(
        #         actions=ins_a, states=ins_s, h=hidden
        # )
        batch_a = next_action_scaled[:, t, ...].unsqueeze(1)
        act_h_ntf = gmm_loss(batch_a, mu_a, sig_a, logpi_a)
        loss_ntf_a += act_h_ntf / ASIZE

        batch_s = next_state_scaled[:, t, ...].unsqueeze(1)
        gmm = gmm_loss(batch_s, mus, sigmas, logpis)
        loss_ntf_s += gmm / LSIZE

        for n in range(NGAUSS):
            print("n", n)
            for m in range(AGAUSS):
                print("m", m, sigmas[(NGAUSS * m) + n].unsqueeze(0).shape)
                ns, _ = worldvae.get_gmm_sample_from_mode(
                    mus[(NGAUSS * m) + n].unsqueeze(0).detach().clone(),
                    sigmas[(NGAUSS * m) + n].unsqueeze(0).detach().clone(),
                    n,
                    False,
                )  # mus should be B, T,G,  F (1, 1, 8, 14)
                print(ns.shape, pred_s.shape)
                print(sigmas.shape)
                pred_s[:, t, (NGAUSS * m) + n, :] = ns  # .squeeze(1)
                print(sig_s_lst[:, t, (NGAUSS * m) + n, :].shape)
                print("t", t)
                print(sigmas[(NGAUSS * m) + n, :, n, :].shape)
                sig_s_lst[:, t, (NGAUSS * m) + n, :] = sigmas[(NGAUSS * m) + n, :, n, :]

                na, _ = worldvae.get_gmm_sample_from_mode(
                    mu_a[(NGAUSS * m) + n].unsqueeze(0).detach().clone(),
                    sig_a[(NGAUSS * m) + n].unsqueeze(0).detach().clone(),
                    m,
                    False,
                )  # mus should be B, T,G,  F (1, 1, 2, 14)
                print("na", na.shape, mu_a.shape)
                pred_act[:, t, (NGAUSS * m) + n, :] = na
                sig_a_lst[:, t, (NGAUSS * m) + n, :] = sig_a[(NGAUSS * m) + n, :, m, :]

        print("pred act", pred_act.shape)
        ins_a = pred_act[:, t, :, :].permute(1, 0, 2)  # B, T, 1, F
        ins_s = pred_s[:, t, :, :].permute(1, 0, 2)

        mse_a_ntf += f.mse_loss(batch_a, na)
        mse_s_ntf += f.mse_loss(batch_s, ns)  # , unstandardize(ns, mean_ns, std_ns))
    # pdb.set_trace()

    # Losses
    loss = (1 - ratio) * (
        loss_tf_a + loss_ntf_a + loss_tf_s + loss_ntf_s
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
    # print('plotting', pred_act.contiguous().shape, next_action.shape, sig_a_lst.shape)

    plot_mse_act(
        pred_act.contiguous().cpu().numpy(),
        next_action.contiguous().cpu().numpy(),
        epoch,
        train,
        std=sig_a_lst.detach().clone().cpu().numpy(),
    )

    pred_s_unstd = unstandardize(pred_s, mean_ns, std_ns).to(device, non_blocking=True)
    print(
        "plotting", pred_s_unstd.contiguous().shape, next_state.shape, sig_s_lst.shape
    )
    plot_mse_state(
        pred_s_unstd.contiguous().cpu().numpy(),
        next_state.contiguous().cpu().numpy(),
        epoch,
        train,
        std=sig_s_lst.detach().clone().cpu().numpy(),
    )
    pdb.set_trace()

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
