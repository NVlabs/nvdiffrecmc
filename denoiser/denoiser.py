import os

import torch
import numpy as np
import math

from render import util
from render import optixutils as ou

###############################################################################
# Bilateral denoiser
#
# Loosely based on SVGF, but removing temporal components and variance stopping guides.
# https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced
###############################################################################

class BilateralDenoiser(torch.nn.Module):
	def __init__(self, influence=1.0):
		super(BilateralDenoiser, self).__init__()
		self.set_influence(influence)

	def set_influence(self, factor):
		self.sigma = max(factor * 2, 0.0001)
		self.variance = self.sigma**2.
		self.N = 2 * math.ceil(self.sigma * 2.5) + 1

	def forward(self, input):
		col    = input[..., 0:3]
		nrm    = util.safe_normalize(input[..., 3:6]) # Bent normals can produce normals of length < 1 here
		zdz    = input[..., 6:8]
		return ou.bilateral_denoiser(col, nrm, zdz, self.sigma)
