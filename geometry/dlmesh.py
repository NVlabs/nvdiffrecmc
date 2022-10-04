# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from render import mesh
from render import render
from render import regularizer
import render.optixutils as ou

###############################################################################
#  Geometry interface
###############################################################################

class DLMesh:
    def __init__(self, initial_guess, FLAGS):
        self.FLAGS = FLAGS

        self.initial_guess     = initial_guess
        self.mesh              = initial_guess.clone()

        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()

        self.mesh.v_pos.requires_grad_(True)

        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        print("Avg edge length: %f" % regularizer.avg_edge_length(self.mesh.v_pos, self.mesh.t_pos_idx))

    def parameters(self):
        return [self.mesh.v_pos]

    def getOptimizer(self, lr_pos):
        return torch.optim.Adam(self.parameters(), lr=lr_pos)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)

        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, FLAGS, denoiser):
        
        color_ref = target['img']

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        opt_mesh = self.getMesh(opt_material)
        buffers = render.render_mesh(FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], target['light'] if lgt is None else lgt, target['resolution'],
                                       spp=target['spp'], num_layers=FLAGS.layers, msaa=True, background=target['background'], 
                                       optix_ctx=self.optix_ctx, denoiser=denoiser)

        t_iter = iteration / FLAGS.iter

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        # Image-space loss, split into a coverage component and a color component
        img_loss  = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Monochrome shading regularizer
        reg_loss += regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, FLAGS.lambda_diffuse, FLAGS.lambda_specular)

        # Material smoothness regularizer
        reg_loss += regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)

        # Chroma regularizer
        reg_loss += regularizer.chroma_loss(buffers['kd'], color_ref, self.FLAGS.lambda_chroma)

        # Perturbed normal regularizer
        if 'perturbed_nrm_grad' in buffers:
            reg_loss += torch.mean(buffers['perturbed_nrm_grad']) * FLAGS.lambda_nrm2

        # Laplacian regularizer. 
        if self.FLAGS.laplace == "absolute":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * FLAGS.laplace_scale * (1 - t_iter)
        elif self.FLAGS.laplace == "relative":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * FLAGS.laplace_scale * (1 - t_iter)                

        return img_loss, reg_loss