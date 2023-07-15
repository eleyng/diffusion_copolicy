import pdb
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
import numpy as np


def gmm_loss(
    batch, mus, sigmas, logpi, reduce=True
):  # pylint: disable=too-many-arguments
    """Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    # print('b', batch.size(), mus.size())
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob


class _MDRNNBase(nn.Module):
    def __init__(
        self,
        states,
        actions,
        hiddens,
        gaussians,
        gaussians_a,
        batch_size,
        n_layers,
        device,
        n_bins,
        control_type,
    ):
        super().__init__()
        self.device = device
        self.states = states
        self.actions = actions
        self.features = self.states  # TODO: change if diff
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.gaussians_a = gaussians_a
        self.batch_size = batch_size
        self.num_layers = n_layers
        self.n_bins = n_bins
        self.control_type = control_type

        self.relu = nn.LeakyReLU()  # Rewards

    def forward(self, *inputs):
        pass


class MDRNN(_MDRNNBase):
    """MDRNN model for multi steps forward"""

    def __init__(
        self,
        states,
        actions,
        hiddens,
        gaussians,
        gaussians_a,
        batch_size,
        n_layers,
        device,
        n_bins,
        control_type,
    ):
        super().__init__(
            states,
            actions,
            hiddens,
            gaussians,
            gaussians_a,
            batch_size,
            n_layers,
            device,
            n_bins,
            control_type,
        )

        assert n_bins is not None, "Did not set the number of classes, n_bins"
        emb_i = states + actions
        emb_o = 32
        m = 1
        gru_in = states
        c = 0  # states + actions
        emb_a = m * hiddens

        """self.emb = nn.Sequential(
                        nn.Linear(emb_i, emb_a),
                        nn.ReLU(),
                        nn.Linear(m*hiddens, m*hiddens),
                        nn.ReLU(),
                        )"""
        # self.emb_a = nn.Embedding(n_bins, emb_a)
        self.emb = nn.Linear(emb_i, emb_o)
        # self.gru = nn.GRU(gru_in, m*hiddens, n_layers, batch_first=True)
        self.gru_s = nn.GRU(emb_o, m * hiddens, n_layers, batch_first=True)
        self.gru_a = nn.GRU(emb_o, m * hiddens, n_layers, batch_first=True)
        # self.gru_r = nn.GRU(2, m*hiddens, n_layers)
        # self.rewards_linear = nn.Sequential(
        #            nn.Linear(m*hiddens + c, 1),
        #            nn.ReLU(),
        #            )
        self.trick = True  # False for test time -- TODO: set False in Cell network

        # Hardmaru MDN
        self.z_pi = nn.Linear(m * hiddens + c, gaussians)
        self.z_sigma = nn.Linear(m * hiddens + c, self.states * gaussians)
        self.z_mu = nn.Linear(m * hiddens + c, self.states * gaussians)

        # Categorical Mixture Model Network
        # Hardmaru MDN
        if control_type == "discrete":
            if n_bins is not None:
                self.n_bins = n_bins
            else:
                self.n_bins = 1
            self.z_pi_c = nn.Linear(m * hiddens + c, self.n_bins)
        else:
            self.z_pi_a = nn.Linear(m * hiddens + c, self.gaussians_a)
            self.z_sigma_a = nn.Linear(m * hiddens + c, self.actions * self.gaussians_a)
            self.z_mu_a = nn.Linear(m * hiddens + c, self.actions * self.gaussians_a)
            # self.z_pi_a = nn.Sequential(
            #         nn.Linear(m*hiddens, m*hiddens),
            #         nn.LeakyReLU(),
            #         nn.Linear(m*hiddens, self.gaussians_a),
            #         nn.LeakyReLU(),
            #         )
            # self.z_sigma_a = nn.Sequential(
            #         nn.Linear(m*hiddens, m*hiddens),
            #         nn.LeakyReLU(),
            #         nn.Linear(m*hiddens, self.actions * self.gaussians_a),
            #         nn.LeakyReLU(),
            #         )
            # self.z_mu_a = nn.Sequential(
            #         nn.Linear(m*hiddens, m*hiddens),
            #         nn.LeakyReLU(),
            #         nn.Linear(m*hiddens, self.actions * self.gaussians_a),
            #         nn.LeakyReLU(),
            #         )

    def init_hidden(self, random=False):
        if random:
            return torch.randn(
                self.num_layers, self.batch_size, self.hiddens, device=self.device
            )
        else:
            return torch.zeros(
                self.num_layers, self.batch_size, self.hiddens, device=self.device
            )

    def get_gmm_sample(self, mus, sigmas, pis, trick=False):
        """Sample from GMM"""

        j = Categorical(pis).sample().unsqueeze(-1)
        sampled_mus = torch.zeros(mus.size(0), mus.size(1), mus.size(-1)).to(
            self.device
        )
        sampled_sigmas = torch.zeros(mus.size(0), mus.size(1), mus.size(-1)).to(
            self.device
        )
        # index into mus and sigmas with j
        for t in range(mus.size(1)):
            for b in range(mus.size(0)):
                idx = j[b, t, :]
                sampled_mus[b, t, :] = mus[b, t, idx, :]
                sampled_sigmas[b, t, :] = sigmas[b, t, idx, :]

        # test time: use mus only
        if trick:
            sampled_lat = sampled_mus
        # train time: reparam trick
        else:
            eps = torch.randn_like(sampled_sigmas).to(self.device)
            sampled_lat = eps.mul(sampled_sigmas).add_(sampled_mus)
            # sampled_lat = torch.normal(sampled_mus, sampled_sigmas).to(self.device)
        return sampled_lat, idx

    def get_gmm_sample_from_mode(self, mus, sigmas, idx, trick=False):
        """Sample from GMM"""

        sampled_mus = torch.zeros(mus.size(0), mus.size(1), mus.size(-1)).to(
            self.device
        )
        sampled_sigmas = torch.zeros(mus.size(0), mus.size(1), mus.size(-1)).to(
            self.device
        )
        # index into mus and sigmas with j
        for t in range(mus.size(1)):
            for b in range(mus.size(0)):
                sampled_mus[b, t, :] = mus[b, t, idx, :]
                sampled_sigmas[b, t, :] = sigmas[b, t, idx, :]

        # test time: use mus only
        if trick:
            sampled_lat = sampled_mus
        # train time: reparam trick
        else:
            eps = torch.randn_like(sampled_sigmas).to(self.device)
            sampled_lat = eps.mul(sampled_sigmas).add_(sampled_mus)
            # sampled_lat = torch.normal(sampled_mus, sampled_sigmas).to(self.device)
        return sampled_lat, idx

    def forward(
        self, actions, states, h=None, h_a=None
    ):  # pylint: disable=arguments-differ
        """MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args states: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next state, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """

        # embed actions, then concat to states for further embed
        seq = actions.size(1)
        # actions = self.emb_a(actions).squeeze()
        # print(states.shape, actions.shape)
        # ins = actions
        ins = torch.cat([states, actions], dim=-1)  # I
        # print('sates, act', states.shape, actions.shape)
        ins = self.emb(ins)
        # ins = states

        # lstm - states
        if h is None:
            outs, hiddens = self.gru_s(ins)
        else:
            # print('hiddens', h.shape, ins.shape)
            outs, hiddens = self.gru_s(ins, h)

        # Densenet
        # outs = torch.cat([outs, ins], dim=-1)

        # GMM - states
        pi = self.z_pi(outs)
        pi = pi.view(-1, seq, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)
        # print('pi', torch.exp(logpi[0, 0, :]))
        logsigmas = self.z_sigma(outs)
        sigmas = torch.exp(logsigmas)
        sigmas = sigmas.view(-1, seq, self.gaussians, self.states)
        mus = self.z_mu(outs)
        mus = mus.view(-1, seq, self.gaussians, self.states)

        # rewards
        # rs = self.rewards_linear(outs)

        # lstm - states
        ins_a = ins
        if h_a is None:
            outs_a, hiddens_a = self.gru_a(ins_a)
        else:
            # print('hiddens', h.shape, ins.shape)
            outs_a, hiddens_a = self.gru_a(ins_a, h_a)

        if self.control_type == "discrete":
            # Categorical distribution - actions
            logits = self.z_pi_c(states)
            logits = logits.view(seq_len * bs, self.n_bins)  # logits
            log_probs = f.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

        else:
            # GMM - actions
            # print('SEQ', seq)
            pi_a = self.z_pi_a(outs_a)
            pi_a = pi_a.view(-1, seq, self.gaussians_a)
            logpi_a = f.log_softmax(pi_a, dim=-1)
            # print('pi', torch.exp(logpi[0, 0, :]))
            logsigmas_a = self.z_sigma_a(outs_a)
            sigmas_a = torch.exp(logsigmas_a)
            sigmas_a = sigmas_a.view(-1, seq, self.gaussians_a, self.actions)
            mus_a = self.z_mu_a(outs_a)
            mus_a = mus_a.view(-1, seq, self.gaussians_a, self.actions)  # 16, 1, 2, 4

        # return logits, probs, hiddens
        # return rs
        # print('here', mus.shape, sigmas.shape, logpi.shape, mus_a.shape, sigmas_a.shape, logpi_a.shape )
        return mus, sigmas, logpi, mus_a, sigmas_a, logpi_a, hiddens, hiddens_a
        # return mus, sigmas, logpi, logits, probs, hiddens, rs #, nz, na #, ds
