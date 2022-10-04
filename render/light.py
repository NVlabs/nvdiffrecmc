# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru

######################################################################################
# Monte-carlo sampled environment light with PDF / CDF computation
######################################################################################

class EnvironmentLight:
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        self.mtx = None
        self.base = base

        self.pdf_scale = (self.base.shape[0] * self.base.shape[1]) / (2 * np.pi * np.pi)
        self.update_pdf()

    def xfm(self, mtx):
        self.mtx = mtx

    def parameters(self):
        return [self.base]

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def update_pdf(self):
        with torch.no_grad():
            # Compute PDF
            Y = util.pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
            self._pdf = torch.max(self.base, dim=-1)[0] * torch.sin(Y * np.pi) # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
            self._pdf = self._pdf / torch.sum(self._pdf)

            # Compute cumulative sums over the columns and rows
            self.cols = torch.cumsum(self._pdf, dim=1)
            self.rows = torch.cumsum(self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0)

            # Normalize
            self.cols = self.cols / torch.where(self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols))
            self.rows = self.rows / torch.where(self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows))

    @torch.no_grad()
    def generate_image(self, res):
        texcoord = util.pixel_grid(res[1], res[0])
        return dr.texture(self.base[None, ...].contiguous(), texcoord[None, ...].contiguous(), filter_mode='linear')[0]

######################################################################################
# Load and store
######################################################################################

@torch.no_grad()
def _load_env_hdr(fn, scale=1.0, res=None):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale

    if res is not None:
        texcoord = util.pixel_grid(res[1], res[0])
        latlong_img = torch.clamp(dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0], min=0.0001)

    print("EnvProbe,", latlong_img.shape, ", min/max", torch.min(latlong_img).item(), torch.max(latlong_img).item())
    return EnvironmentLight(base=latlong_img)

@torch.no_grad()
def load_env(fn, scale=1.0, res=None):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale, res)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

@torch.no_grad()
def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight)
    color = light.generate_image([512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):  
    base = torch.rand(base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    l = EnvironmentLight(base.clone().detach().requires_grad_(True))
    return l
      
