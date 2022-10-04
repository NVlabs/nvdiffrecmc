# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch

from render import util
from render import mesh
from render import render
from render import light
import render.optixutils as ou


from dataset import Dataset

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetMesh(Dataset):
    
    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, validate=False, num_validation_frames=200):
        # Init 
        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]
        self.num_validation_frames = num_validation_frames

        print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        print("Build Optix bvh")
        self.optix_ctx = ou.OptiXContext()
        ou.optix_build_bvh(self.optix_ctx, ref_mesh.v_pos, ref_mesh.t_pos_idx.int(), rebuild=1)
        print("Done building OptiX bvh")


        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Pre-randomize a list with finite number of training samples
        if hasattr(FLAGS, 'train_examples') and FLAGS.train_examples is not None:
            self.train_examples = [self._random_scene() for i in range(FLAGS.train_examples)]
       
        self.ref_mesh = mesh.compute_tangents(ref_mesh)
        self.envlight = light.load_env(FLAGS.envlight, scale=FLAGS.env_scale)

    def getMesh(self):
        return self.ref_mesh

    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display
        ang    = (itr / self.num_validation_frames) * np.pi * 2 
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.envlight, self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization
        mv = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.25)

        mvp      = proj_mtx @ mv
        campos   = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.envlight, iter_res, self.FLAGS.spp # Add batch dimension

    def __len__(self):
        return self.num_validation_frames if self.validate else (self.FLAGS.iter + 0) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            mv, mvp, campos, lgt, iter_res, iter_spp = self._rotate_scene(itr)
        else:
            if hasattr(self, 'train_examples'):
                mv, mvp, campos, lgt, iter_res, iter_spp = self.train_examples[itr % len(self.train_examples)]
            else:
                mv, mvp, campos, lgt, iter_res, iter_spp = self._random_scene()

        img = render.render_mesh(self.FLAGS, self.glctx, self.ref_mesh, mvp, campos, lgt, iter_res, spp=iter_spp,
                                num_layers=self.FLAGS.layers, msaa=True, background=None, 
                                optix_ctx=self.optix_ctx)['shaded'] # Post-mixing in background causes a small anti-aliasing error

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'light' : lgt,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : img,
        }

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        out_batch = {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'light' : batch[0]['light'],
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
        }
        return out_batch

