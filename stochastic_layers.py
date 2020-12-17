import torch

import torch.nn as nn

from model_utils import gaussian

class novae_layer(nn.Module):
    # NO AUTOENCODER (variational and non)
    """
        h
        |
        x
    """
    def __init__(self, input_size, tf_size):
        super(novae_layer, self).__init__()
        self.input_size = input_size
        self.tf_size = tf_size

        self.tf_mlp = nn.Linear(input_size, tf_size)
        # self.q_mean2_mlp = nn.Linear(input_size, latent_z_size)
        # self.q_logvar2_mlp = nn.Linear(input_size, latent_z_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()
        tf = self.tf_mlp(inputs) * mask.unsqueeze(-1)

        return tf


class gaussian_layer(nn.Module):
    # NON-VARIATIONAL AUTOENCODER
    """
        h
        |
        z
        |
        x
    """
    def __init__(self, input_size, latent_z_size):
        super(gaussian_layer, self).__init__()
        self.input_size = input_size
        self.latent_z_size = latent_z_size

        self.z_mlp = nn.Linear(input_size, latent_z_size)
        # self.q_mean2_mlp = nn.Linear(input_size, latent_z_size)
        # self.q_logvar2_mlp = nn.Linear(input_size, latent_z_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()
        z = self.z_mlp(inputs) * mask.unsqueeze(-1)

        return z, z, torch.zeros(tuple(z.shape))


class gaussian_flat_layer(nn.Module):
    """
        h
       / \
      y   z
       \ /
        x

    """
    # NON-VARIATIONAL
    def __init__(self, input_size, latent_z_size, latent_y_size):
        super(gaussian_flat_layer, self).__init__()
        self.input_size = input_size
        self.latent_y_size = latent_y_size
        self.latent_z_size = latent_z_size

        self.z_mlp = nn.Linear(input_size, latent_z_size)
        # self.q_logvar_mlp = nn.Linear(input_size, latent_z_size)

        self.y_mlp = nn.Linear(input_size, latent_y_size)
        # self.q_logvar2_mlp = nn.Linear(input_size, latent_y_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()

        z = self.z_mlp(inputs) * mask.unsqueeze(-1)
        # logvar_qs = self.q_logvar_mlp(inputs) * mask.unsqueeze(-1)

        y = self.y_mlp(inputs) * mask.unsqueeze(-1)
        # logvar2_qs = self.q_logvar2_mlp(inputs) * mask.unsqueeze(-1)

        return z, y, z, torch.zeros(tuple(z.shape)), y, torch.zeros(tuple(y.shape))


class gaussian_hier_layer(nn.Module):
    # NON-VARIAITONAL
    """
        h
        |
        y
        |
        z
        |
        x
    """
    def __init__(self, input_size, latent_z_size, latent_y_size):
        super(gaussian_hier_layer, self).__init__()
        self.input_size = input_size
        self.latent_y_size = latent_y_size
        self.latent_z_size = latent_z_size

        self.y_mlp = nn.Linear(input_size, latent_y_size)
        # self.q_logvar2_mlp = nn.Linear(input_size, latent_y_size)

        self.z_mlp = nn.Linear(input_size + latent_y_size, latent_z_size)
        # self.q_logvar_mlp = nn.Linear(input_size + latent_y_size, latent_z_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()

        y = self.y_mlp(inputs) * mask.unsqueeze(-1)
        # logvar2_qs = self.q_logvar2_mlp(inputs)

        gauss_input = torch.cat([inputs, y], -1)
        z = self.z_mlp(gauss_input) * mask.unsqueeze(-1)
        # logvar_qs = self.q_logvar_mlp(gauss_input)

        return z, y, z, torch.zeros(tuple(z.shape)), y, torch.zeros(tuple(y.shape))
