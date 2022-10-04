# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
from render import regularizer
import render.optixutils as ou

###############################################################################
# DMTet utility functions
###############################################################################

triangle_table = torch.tensor([
        [-1, -1, -1, -1, -1, -1],
        [ 1,  0,  2, -1, -1, -1],
        [ 4,  0,  3, -1, -1, -1],
        [ 1,  4,  2,  1,  3,  4],
        [ 3,  1,  5, -1, -1, -1],
        [ 2,  3,  0,  2,  5,  3],
        [ 1,  4,  0,  1,  5,  4],
        [ 4,  2,  5, -1, -1, -1],
        [ 4,  5,  2, -1, -1, -1],
        [ 4,  1,  0,  4,  5,  1],
        [ 3,  2,  0,  3,  5,  2],
        [ 1,  3,  5, -1, -1, -1],
        [ 4,  1,  2,  4,  3,  1],
        [ 3,  0,  4, -1, -1, -1],
        [ 2,  0,  1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device="cuda")

num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device="cuda")
base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device="cuda")
v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))

def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
        order = order.unsqueeze(dim=1)

        a = torch.gather(input=edges_ex2, index=order, dim=1)      
        b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

    return torch.stack([a, b],-1)

def map_uv(faces, face_gidx, max_idx):
    N = int(np.ceil(np.sqrt((max_idx+1)//2)))
    tex_y, tex_x = torch.meshgrid(
        torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
        torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
    )

    pad = 0.9 / N

    uvs = torch.stack([
        tex_x      , tex_y,
        tex_x + pad, tex_y,
        tex_x + pad, tex_y + pad,
        tex_x      , tex_y + pad
    ], dim=-1).view(-1, 2)

    def _idx(tet_idx, N):
        x = tet_idx % N
        y = torch.div(tet_idx, N, rounding_mode='trunc')
        return y * N + x

    tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
    tri_idx = face_gidx % 2

    uv_idx = torch.stack((
        tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
    ), dim = -1). view(-1, 3)

    return uvs, uv_idx

###############################################################################
# marching tetrahedrons (differentiable)
#
# This function is adapted from 
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

def marching_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum>0) & (occ_sum<4)
        occ_sum = occ_sum[valid_tets]

        # find all vertices
        all_edges = tet_fx4[valid_tets][:,base_tet_edges].reshape(-1,2)
        all_edges = sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
        
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
        idx_map = mapping[idx_map] # map edges to verts

        interp_v = unique_edges[mask_edges]
    edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
    edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
    edges_to_interp_sdf[:,-1] *= -1

    denominator = edges_to_interp_sdf.sum(1,keepdim = True)

    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1,6)

    tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table[tetindex]

    # Generate triangle indices
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1, index=triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1, index=triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
    ), dim=0)

    # Get global face index (static, does not depend on topology)
    num_tets = tet_fx4.shape[0]
    tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
    face_gidx = torch.cat((
        tet_gidx[num_triangles == 1]*2,
        torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
    ), dim=0)

    uvs, uv_idx = map_uv(faces, face_gidx, num_tets*2)

    return verts, faces, uvs, uv_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry:
    def __init__(self, grid_res, scale, FLAGS):
        self.FLAGS    = FLAGS
        self.grid_res = grid_res

        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices']).float().to('cuda') * scale
        self.indices  = torch.tensor(tets['indices']).long().to('cuda')
        self.generate_edges()

        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()

        # Random init
        sdf = torch.rand_like(self.verts[:,0]) - 0.1

        self.sdf    = sdf.clone().detach().requires_grad_(True)
        self.deform = torch.zeros_like(self.verts).clone().detach().requires_grad_(True)

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def parameters(self):
        return [self.sdf, self.deform]

    def getOptimizer(self, lr_pos):
        return torch.optim.Adam(self.parameters(), lr=lr_pos)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        verts, faces, uvs, uv_idx = marching_tets(v_deformed, self.sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, FLAGS, denoiser):

        t_iter = iteration / FLAGS.iter
        color_ref = target['img']

        opt_mesh = self.getMesh(opt_material)

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        shadow_ramp = min(iteration / 1750, 1.0)
        if denoiser is not None: denoiser.set_influence(shadow_ramp)
        buffers = render.render_mesh(FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], target['light'] if lgt is None else lgt, target['resolution'],
                                    spp=target['spp'], num_layers=FLAGS.layers, msaa=True, background=target['background'], optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_ramp)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        # Image-space loss, split into a coverage component and a color component
        img_loss  = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        # SDF regularizer
        sdf_weight = FLAGS.sdf_regularizer - (FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight 

        # Monochrome shading regularizer
        reg_loss += regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, FLAGS.lambda_diffuse, FLAGS.lambda_specular)
        
        # Material smoothness regularizer
        reg_loss += regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)

        # Chroma regularizer
        reg_loss += regularizer.chroma_loss(buffers['kd'], color_ref, self.FLAGS.lambda_chroma)
        assert 'perturbed_nrm' not in buffers # disable normal map in first pass

        return img_loss, reg_loss

