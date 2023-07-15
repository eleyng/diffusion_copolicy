import pdb
import torch
import yaml
import gym
import time
import pygame
import numpy as np
from os.path import join, exists, dirname, abspath
from os import mkdir
from stable_baselines3.common.env_checker import check_env
import sys

sys.path.insert(1, dirname(abspath(__file__)) + "/../models")
from bc import sample_discrete, BCRNN
from mdrnn import MDRNN

FPS = 30
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS


torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
yaml_filepath = join(dirname(__file__), "config/test-a.yml")
parser = argparse.ArgumentParser("WorldVAE training & validation")
parser.add_argument(
    "--logdir", type=str, help="Where things are logged and models are loaded from."
)
parser.add_argument(
    "--noreload", action="store_true", help="Do not reload if specified."
)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
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
teacher_forcing_ratio = cfg["teacher_forcing"]
freq = cfg["freq"]
epochs = cfg["epochs"]
time_limit = cfg["time_limit"]
beta_min = cfg["beta_min"]
beta_max = cfg["beta_max"]

# Initialize other parameters
b_step = 4
counter = 0
beta_interval = 1000
max_beta_int = beta_interval + 2000
min_beta_int = beta_interval
beta = beta_min


# Create dataloader + dataset
data = dict(np.load(human_data))


# Get training data statistics for standardizations
train_stats = dict(np.load(join(data_base, data_dir, train_stats_f)))
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
worldvae_dir = join(data_base, args.logdir, experiment_name)
worldvae_file = join(worldvae_dir, "best.tar")
if not exists(worldvae_dir):
    mkdir(worldvae_dir)
# Create samples directory
sample_dir = join(data_base, worldvae_dir, "samples")
if not exists(sample_dir):
    mkdir(sample_dir)
if not exists(join(sample_dir, "train")):
    mkdir(join(sample_dir, "train"))
    for b in range(0, BSIZE, b_step):
        mkdir(join(join(sample_dir, "train"), "batch-{}".format(str(b))))
if not exists(join(sample_dir, "valid")):
    mkdir(join(sample_dir, "valid"))
    for b in range(0, BSIZE, b_step):
        mkdir(join(join(sample_dir, "valid"), "batch-{}".format(str(b))))

# Initialize dynamical model
worldvae = MDRNN(
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
)
worldvae.to(device, non_blocking=True)

# Initialize training parameters
optimizer = torch.optim.RMSprop(
    worldvae.parameters(), lr=1e-3, alpha=0.9, weight_decay=0.2
)
# optimizer = torch.optim.Adam(worldvae.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=20)
# earlystopping = EarlyStopping('min', patience=30)

# Load dynamical model parameters
if exists(worldvae_file) and not args.noreload:
    worldvae_state = torch.load(
        worldvae_file
    )  # , map_location=torch.device('cpu')) if device == 'cpu' else torch.load(vae_file)
    print(
        "Loading WorldVAE at epoch {} "
        "with test error {}".format(
            worldvae_state["epoch"], worldvae_state["precision"]
        )
    )
    state_dict = worldvae_state["state_dict"]
    state_dict2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if "_l0" in key:
            name = key.split("_")
            state_dict2["_".join(name[:-1])] = state_dict2.pop(key)
    if mode == "rollout":
        worldvae.load_state_dict(state_dict2)
    else:
        worldvae.load_state_dict(state_dict)
    optimizer.load_state_dict(worldvae_state["optimizer"])
    scheduler.load_state_dict(worldvae_state["scheduler"])
    # earlystopping.load_state_dict(worldvae_state['earlystopping'])


def reset_hidden():
    return torch.zeros(2, 1, 2 * RSIZE)


# Starting env
env = gym.make("cooperative_transport.gym_table:table-v0")
# print("Checking environment!")
# check_env(env, warn=True)

obs = env.reset()
hidden = reset_hidden()
hidden = hidden.repeat(1, K, 1).float()
u = torch.tensor(0.0).to(device)
n_steps = 100001
n_iter = 0
running = True
env_done = False
next_game_tick = time.time()
clock = pygame.time.Clock()

while running and n_iter < n_steps:
    loops = 0
    if env_done:
        print("Done.")
        pygame.quit()
        break
    else:
        while time.time() > next_game_tick and loops < MAX_FRAMESKIP:
            # Standardize obs
            obs = env.standardize(torch.tensor(obs[:6]), mean_s, std_s)
            with torch.no_grad():
                if n_iter % SEQ_LEN == 0 and n_iter != 0:
                    hidden = reset_hidden()
                    hidden = hidden.repeat(1, K, 1).float()

                    print("Reinitialize hidden state: ", hidden)
                # action = self.standardize(action, mean_a, std_a)
                # Run MPC #TODO: keep params for config
                u = (
                    u.repeat(K, 1).unsqueeze(0).type(torch.LongTensor).to(device)
                )  # T, K, F
                obs = obs.repeat(K, 1).unsqueeze(0)  # T, K, F
                cum_r = np.zeros((1, K, 1))  # T, K, F
                for h in range(H):
                    mus, sigmas, logpi, logits, probs, hidden, rs = model(
                        u, obs.float(), hidden
                    )
                    if h == 1:
                        tmp_hidden = hidden
                        tmp_action = u
                    cum_r = np.add(cum_r, rs.detach().numpy())
                    obs, _ = model.get_gmm_sample(mus, sigmas, torch.exp(logpi), True)
                    probs = probs.view(1, K, -1)
                    u = torch.argmax(probs, dim=2).unsqueeze(-1)

                cum_r = cum_r.squeeze()
                print(cum_r)
                best_idx = np.where(cum_r == cum_r.max())
                u = u[:, best_idx[0][0], :].squeeze()
                # Updates
                hidden = tmp_hidden[:, best_idx[0][0], :].unsqueeze(1)
                hidden = hidden.repeat(1, K, 1).float()
                obs, reward, done, info = env.step(u, CONST_DT)
                print("loop: ", loops, info)
                next_game_tick += CONST_DT
                loops += 1

            # UPDATE DISPLAY
            env.redraw()
            ##if callback is not None:
            ##    callback(prev_obs, obs, action, rew, env_done, info)
            # CLOCK TICK
            clock.tick(FPS)
            if clock.get_fps() > 0:
                print("Reported dt: ", 1 / clock.get_fps())
        if obs is not None:
            rendered = env.render(mode="rgb_array")
        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key == 27:
                    running = False
            elif event.type == pygame.QUIT:
                running = False
        n_iter += 1
