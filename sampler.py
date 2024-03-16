import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray)
        z_vals = torch.arange(start=self.min_depth, end=self.max_depth, step=(self.max_depth-self.min_depth)/self.n_pts_per_ray).to(ray_bundle.directions.device)

        # TODO (Q1.4): Sample points from z values
        # N, D = ray_bundle.origins.shape[0], z_vals.shape[0]
        # origins = 
        # sample_points = ray_bundle.origins + ray_bundle.directions * z_vals[..., None]

        # Return
        return ray_bundle._replace(
            sample_points=ray_bundle.origins.unsqueeze(1) + torch.einsum('mi,n->mni', ray_bundle.directions, z_vals),
            sample_lengths=torch.tile(z_vals,(ray_bundle.directions.shape[0],1)).unsqueeze(2) ,
        )

class HierarchicalStratifiedSampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
        fine = False,
        weights = None
    ):
        if not fine:
            # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
            z_range = torch.arange(start=self.min_depth, end=self.max_depth, step=(self.max_depth-self.min_depth)/self.n_pts_per_ray).to(ray_bundle.directions.device)

        else:
            if weights is None:
                weights = torch.ones_like(ray_bundle.directions)
            weights += 1e-6
            pdf = weights / weights.sum(1, keepdim=True)
            cdf = torch.cumsum(pdf, -1)
            u = torch.rand_like(cdf[..., :1])
            z_range = torch.searchsorted(cdf, u, right=True).squeeze(-1).float()
            z_range = z_range * (self.max_depth - self.min_depth) / self.n_pts_per_ray + self.min_depth

        return ray_bundle._replace(
            sample_points=ray_bundle.origins.unsqueeze(1) + torch.einsum('mi,n->mni', ray_bundle.directions, z_range),
            sample_lengths=torch.tile(z_range,(ray_bundle.directions.shape[0],1)).unsqueeze(2) ,
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}