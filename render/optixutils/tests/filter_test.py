# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pickletools import read_float8
import torch

import os
import sys
import math
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import optixutils as ou
import numpy as np

RES = 1024
DTYPE = torch.float32

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
	return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
	return x / length(x, eps)

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	return torch.sum(x*y, -1, keepdim=True)

class BilateralDenoiser(torch.nn.Module):
	def __init__(self, sigma=1.0):
		super(BilateralDenoiser, self).__init__()
		self.set_sigma(sigma)

	def set_sigma(self, sigma):
		self.sigma = max(sigma, 0.0001)
		self.variance = self.sigma**2.
		self.N = 2 * math.ceil(self.sigma * 2.5) + 1

	def forward(self, input):
		eps    = 0.0001
		col    = input[..., 0:3]
		nrm    = input[..., 3:6]
		kd     = input[..., 6:9]
		zdz    = input[..., 9:11]

		accum_col = torch.zeros_like(col)
		accum_w = torch.zeros_like(col[..., 0:1])
		for y in range(-self.N, self.N+1):
			for x in range(-self.N, self.N+1):

				ty, tx = torch.meshgrid(torch.arange(0, input.shape[1], dtype=torch.float32, device="cuda"), torch.arange(0, input.shape[2], dtype=torch.float32, device="cuda"))
				tx = tx[None, ..., None] + x
				ty = ty[None, ..., None] + y

				dist_sqr = (x**2 + y**2)
				dist = np.sqrt(dist_sqr)
				w_xy = np.exp(-dist_sqr / (2 * self.variance))

				with torch.no_grad():
					nrm_tap = torch.roll(nrm, (-y, -x), (1, 2))
					w_normal = torch.pow(torch.clamp(dot(nrm_tap, nrm), min=eps, max=1.0), 128.0)           # From SVGF

					zdz_tap = torch.roll(zdz, (-y, -x), (1, 2))
					w_depth = torch.exp(-(torch.abs(zdz_tap[..., 0:1] - zdz[..., 0:1]) / torch.clamp(zdz[..., 1:2] * dist, min=eps)) ) # From SVGF	

					w = w_xy * w_normal * w_depth
					w = torch.where((tx >= 0) & (tx < input.shape[2]) & (ty >= 0) & (ty < input.shape[1]), w, torch.zeros_like(w))

				col_tap = torch.roll(col, (-y, -x), (1, 2))
				accum_col += col_tap * w
				accum_w += w
		return accum_col / torch.clamp(accum_w, min=eps)

def relative_loss(name, ref, cuda):
	ref = ref.float()
	cuda = cuda.float()
	denom = torch.where(ref > 1e-7, ref, torch.ones_like(ref))
	relative = torch.abs(ref - cuda) / denom
	print(name, torch.max(relative).item())


def test_filter():
	img_cuda = torch.rand(1, RES, RES, 11, dtype=DTYPE, device='cuda')
	img_cuda[..., 3:6] = safe_normalize(img_cuda[..., 3:6])
	img_ref = img_cuda.clone().detach().requires_grad_(True)
	img_cuda = img_cuda.clone().detach().requires_grad_(True)
	target_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	target_ref = target_cuda.clone().detach().requires_grad_(True)
	
	SIGMA = 2.0

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	start.record()
	denoiser = BilateralDenoiser(sigma=SIGMA)
	denoised_ref = denoiser.forward(img_ref)
	ref_loss = torch.nn.MSELoss()(denoised_ref, target_ref)
	ref_loss.backward()
	end.record()
	torch.cuda.synchronize()
	print("Python:", start.elapsed_time(end))

	start.record()
	denoised_cuda = ou.svgf(img_cuda[..., 0:3], img_cuda[..., 3:6], img_cuda[..., 9:11], img_cuda[..., 6:9], SIGMA)
	cuda_loss = torch.nn.MSELoss()(denoised_cuda, target_cuda)
	cuda_loss.backward()
	end.record()
	torch.cuda.synchronize()
	print("CUDA:", start.elapsed_time(end))

	print("-------------------------------------------------------------")
	print("    Filter loss:")
	print("-------------------------------------------------------------")

	relative_loss("denoised:", denoised_ref[..., 0:3], denoised_cuda[..., 0:3])
	relative_loss("grad:", img_ref.grad[..., 0:3], img_cuda.grad[..., 0:3])

test_filter()