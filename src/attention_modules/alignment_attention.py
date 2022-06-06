"""
This file contains source code for the implementation of alignment attention proposed by Zheng et al. 2022 in
Alignment Attention by Matching Key and Query Distributions (https://arxiv.org/abs/2110.12567).
The current version of alignment attention utilizes conditional transport proposed by Zheng et al. 2020 in
Exploiting Chain Rule and Bayes' Theorem to Compare obability Distributions (https://arxiv.org/abs/2012.14100). The
implemented code was inspired by https://github.com/JegZheng/CT-pytorch.

    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 06/03/2022
    @last modified: 06/03/2022
"""

# installed libraries
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlignmentAttention(nn.Module):
    """
    Alignment attention following Zheng et al. 2022 and the conditional transport implementation of Zheng et al. 2020.

    Parameters
    ----------
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    dim: int; default=256
        Projection dimension of Critic and Navigator Networks
    nav_hidden : int; default=512
        Hidden dimension of navigator class hidden layers
    rho : float; default=0.5
        balance coefficient of forward-backward

    """
    def __init__(self,
                 latent_doc_dim: int,
                 dim: int = 256,
                 nav_hidden: int = 512,
                 rho: float = 0.5):
        super().__init__()
        self._latent_doc_dim = latent_doc_dim
        self._dim = dim
        self._nav_hidden = nav_hidden
        self._rho = rho

        # Init Critic and Navigator network
        self._critic = Critic(latent_doc_dim=self._latent_doc_dim,
                              dim=self._dim)
        self._critic.to(device=device)
        self._critic = torch.nn.DataParallel(self._critic)
        self._navigator = Navigator(dim=self._dim,
                                    hidden=self._nav_hidden)
        self._navigator.to(device=device)
        self._navigator = torch.nn.DataParallel(self._navigator)
        self._optim_critic = torch.optim.AdamW(self._critic.parameters(), lr=0.0001, betas=(0.0, 0.9))
        self._optim_navigator = torch.optim.AdamW(self._navigator.parameters(), lr=0.0001, betas=(0.0, 0.9))

        #self.scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(self._optim_critic, gamma=0.99)
        #self.scheduler_navigator = torch.optim.lr_scheduler.ExponentialLR(self._optim_navigator, gamma=0.99)

    def forward(self, K: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the alignment attention module
        Parameters
        ----------
        K : torch.Tensor
            Key matrix with shape [batch_size, number_queries, embedding_dim]
        Q : torch.Tensor
            Query matrix with shape [batch_size, sequence_length, embedding_dim]

        Returns
        -------
        torch.Tensor

        """
        # Need to make sure that Q and K have same dimension
        Q = torch.unsqueeze(Q, dim=0).repeat(K.size()[0], 1, 1)

        f_K = self._critic(K)  # feature of K: b x d
        f_Q = self._critic(Q)  # feature of Q: b x d
        cost = torch.norm(f_K[:, None] - f_Q, dim=-1).pow(2).clone()  # pairwise cost: B x B

        ######################## compute transport map ######################
        mse_n = torch.pow((f_K[:, None] - f_Q), exponent=2)  # pairwise mse for navigator network: B x B x d
        dist = torch.mul(torch.squeeze(self._navigator(mse_n)), -1)
        #dist = self._navigator(mse_n).squeeze().mul(-1).clone()  # navigator distance: B x B
        forward_map = torch.softmax(dist, dim=1)  # forward map is in y wise
        backward_map = torch.softmax(dist, dim=0)  # backward map is in x wise

        ######################## compute CT loss ######################
        # element-wise product of cost and transport map
        CT_loss = self._rho * torch.mean(torch.sum((cost * forward_map), dim=1)) + (1 - self._rho) * torch.mean(torch.sum((cost * backward_map), dim=0))
        print(f"CT-Loss: {CT_loss}")
        return CT_loss


class Critic(nn.Module):
    """
    Critic class to compute the point-to-point cost as cη(q, k) = 1 − τη(k)T τη(q) ||τη(k)||2||τη(q)||2 ,
      where τη(·) is a neural network based “critic” function whose parameter. The critic architecture utilizes the
      highway architecture proposed by Srivastava et al. 2020 in Training Very Deep Networks.
    """
    def __init__(self,
                 latent_doc_dim: int,
                 dim: int = 256):
        super().__init__()
        self._latent_doc_dim = latent_doc_dim
        self._dim = dim

        # Init transform gate: T = Sigmoid(FC(X, W_T))
        self.fc_T = nn.Linear(in_features=self._latent_doc_dim, out_features=self._latent_doc_dim)
        nn.init.xavier_uniform_(self.fc_T.weight)
        self.fc_T.bias.data.fill_(-1)
        self.act_T = nn.Sigmoid()

        # Init general transformation
        self.fc_H = nn.Linear(in_features=self._latent_doc_dim, out_features=self._latent_doc_dim)
        nn.init.xavier_uniform_(self.fc_H.weight)
        self.fc_H.bias.data.fill_(0.01)
        self.act_H = nn.ReLU()

        # Project highway output to a vector space
        self.fc1 = nn.Linear(in_features=self._latent_doc_dim, out_features=self._dim)
        self.fc2 = nn.Linear(in_features=self._dim, out_features=self._dim)
        self.act_fc = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Parameters
        ----------
        x : torch.Tensor
            Either key or query matrix with shape [batch_size, sequence_length, embedding_dim]

        Returns
        -------
        torch.Tensor
            Shape: [batch_size, d], where d = sequence_length x embedding_dim

        """
        # Compute transform gate
        tau = self.act_T(self.fc_T(x))
        # Compute H output
        H = self.act_H(self.fc_H(x))

        # Compute output of transform and carry gates
        I_transform = torch.mul(H, tau)

        I_carry = torch.mul(x, (1 - tau))

        # compute output of highway architecture I
        I = I_transform + I_carry
        I = torch.sum(I.permute(0, 2, 1), dim=-1)

        out = self.fc2(self.act_fc(self.fc1(I)))
        return out


class Navigator(nn.Module):
    """
    Class that transforms the multidimensional input to 1 value.

    Parameters
    ----------
    dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    hidden : int; default=512
        Hidden dimension of the linear layers

    """
    def __init__(self,
                 dim: int = 256,
                 hidden: int = 512):
        super().__init__()
        self._dim = dim
        self._hidden = hidden

        # init fully connected layers
        self.fc1 = nn.Linear(in_features=self._dim, out_features=self._hidden)
        self.fc2 = nn.Linear(in_features=self._hidden, out_features=self._hidden // 2)
        self.fc3 = nn.Linear(in_features=self._hidden // 2, out_features=1)

        # init activation function
        self.LeakyRelu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Navigator class in alignment attention.

        Parameters
        ----------
        x : torch.Tensor
            Either key or query matrix with shape [batch_size, sequence_length, embedding_dim]

        Returns
        -------
        torch.Tensor
            Shape: 1-dimensional element

        """
        x = self.LeakyRelu(self.fc1(x))
        x = self.LeakyRelu(self.fc2(x))
        m = self.fc3(x).clone()
        return m
