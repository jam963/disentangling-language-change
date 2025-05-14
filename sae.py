#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k

    def forward(self, x):
        h = self.encoder(x)

        topk, _ = torch.topk(h, self.k, dim=1)
        mask = h >= topk[:, [-1]]
        h_sparse = h * mask.float()

        return self.decoder(h_sparse), h_sparse, h

class SAELoss(nn.Module):
    def __init__(self, n_latents, k_active, k_aux=None, aux_scale=1/32, samples_per_epoch=None):
        super().__init__()
        self.n_latents = n_latents
        self.k_active = k_active
        self.k_aux = k_aux if k_aux is not None else 2 * k_active
        self.aux_scale = aux_scale
        self.samples_per_epoch = samples_per_epoch
        self.latent_activations = torch.zeros(n_latents)
        self.samples_processed = 0

    def to(self, device):
        super().to(device)
        self.latent_activations = self.latent_activations.to(device)
        return self

    def update_dead_latents(self, h_sparse):
        self.latent_activations += (h_sparse.abs().sum(dim=0) > 0).float()
        self.samples_processed += h_sparse.shape[0]

        if self.samples_processed >= self.samples_per_epoch:
            self.dead_latents = self.latent_activations == 0
            self.latent_activations.zero_()
            self.samples_processed = 0

    def forward(self, x, x_recon, h_sparse, encoder, decoder):
        batch_size = x.shape[0]

        mse_loss = F.mse_loss(x_recon, x)

        self.update_dead_latents(h_sparse)

        aux_loss = torch.tensor(0.0, device=x.device)
        if hasattr(self, 'dead_latents') and self.dead_latents.any():
            # Reconstruction error
            e = x - x_recon

            # Create sparse representation for dead latents
            z = torch.zeros_like(h_sparse)
            z[:, self.dead_latents] = encoder(x)[:, self.dead_latents]

            # Compute k_aux, ensure not bigger than number of dead latents
            k = min(self.k_aux, self.dead_latents.sum().item(), z.size(1))

            if k > 0:
                # Top-k sparsity only if enough dead latents
                values , _ = torch.topk(z[:, self.dead_latents], k, dim=1)
                kth_values = values[:, -1].unsqueeze(1)
                mask = (z[:, self.dead_latents] >= kth_values).float()
                z[:, self.dead_latents] *= mask

            e_hat = decoder(z)

            aux_loss = F.mse_loss(e_hat, e)

        total_loss = mse_loss + self.aux_scale * aux_loss

        return total_loss, mse_loss, aux_loss
