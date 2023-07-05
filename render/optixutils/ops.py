# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import sys
import torch
import torch.utils.cpp_extension

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_plugin = None
if _plugin is None:

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        optix_include_dir = os.path.dirname(__file__) + r"\include"

        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                vs_editions = glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition) \
                    + glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition)
                paths = sorted(vs_editions, reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    elif os.name == 'posix':
        optix_include_dir = os.path.dirname(__file__) + r"/include"

    include_paths = [optix_include_dir]

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/denoising.cu',
        'c_src/optix_wrapper.cpp',
        'c_src/torch_bindings.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='optixutils_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, extra_include_paths=include_paths, with_cuda=True, verbose=True)

    # Import, cache, and return the compiled module.
    import optixutils_plugin
    _plugin = optixutils_plugin

#----------------------------------------------------------------------------
# OptiX autograd func
#----------------------------------------------------------------------------

class _optix_env_shade_func(torch.autograd.Function):
    _random_perm = {}

    @staticmethod
    def forward(ctx, optix_ctx, mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, BSDF, n_samples_x, rnd_seed, shadow_scale):
        _rnd_seed = np.random.randint(2**31) if rnd_seed is None else rnd_seed
        if n_samples_x not in _optix_env_shade_func._random_perm:
            # Generate (32k) tables with random permutations to decorrelate the BSDF and light stratified samples
            _optix_env_shade_func._random_perm[n_samples_x] = torch.argsort(torch.rand(32768, n_samples_x * n_samples_x, device="cuda"), dim=-1).int()

        diff, spec = _plugin.env_shade_fwd(optix_ctx.cpp_wrapper, mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, _optix_env_shade_func._random_perm[n_samples_x], BSDF, n_samples_x, _rnd_seed, shadow_scale)
        ctx.save_for_backward(mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols)
        ctx.optix_ctx = optix_ctx
        ctx.BSDF = BSDF
        ctx.n_samples_x = n_samples_x
        ctx.rnd_seed = rnd_seed
        ctx.shadow_scale = shadow_scale
        return diff, spec
    
    @staticmethod
    def backward(ctx, diff_grad, spec_grad):
        optix_ctx = ctx.optix_ctx
        _rnd_seed = np.random.randint(2**31) if ctx.rnd_seed is None else ctx.rnd_seed
        mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols = ctx.saved_variables
        gb_pos_grad, gb_normal_grad, gb_kd_grad, gb_ks_grad, light_grad = _plugin.env_shade_bwd(
            optix_ctx.cpp_wrapper, mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, _optix_env_shade_func._random_perm[ctx.n_samples_x], 
            ctx.BSDF, ctx.n_samples_x, _rnd_seed, ctx.shadow_scale, diff_grad, spec_grad)
        return None, None, None, gb_pos_grad, gb_normal_grad, None, gb_kd_grad, gb_ks_grad, light_grad, None, None, None, None, None, None, None

class _bilateral_denoiser_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, zdz, sigma):
        ctx.save_for_backward(col, nrm, zdz)
        ctx.sigma = sigma
        out = _plugin.bilateral_denoiser_fwd(col, nrm, zdz, sigma)
        return out
    
    @staticmethod
    def backward(ctx, out_grad):
        col, nrm, zdz = ctx.saved_variables
        col_grad = _plugin.bilateral_denoiser_bwd(col, nrm, zdz, ctx.sigma, out_grad)
        return col_grad, None, None, None

#----------------------------------------------------------------------------
# OptiX ray tracing utils
#----------------------------------------------------------------------------

class OptiXContext:
    def __init__(self):
        print("Cuda path", torch.utils.cpp_extension.CUDA_HOME)
        self.cpp_wrapper = _plugin.OptiXStateWrapper(os.path.dirname(__file__), torch.utils.cpp_extension.CUDA_HOME)

def optix_build_bvh(optix_ctx, verts, tris, rebuild):
    assert tris.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert verts.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    _plugin.optix_build_bvh(optix_ctx.cpp_wrapper, verts.view(-1, 3), tris.view(-1, 3), rebuild)

def optix_env_shade(optix_ctx, mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, BSDF='pbr', n_samples_x=8, rnd_seed=None, shadow_scale=1.0):
    iBSDF = ['pbr', 'diffuse', 'white'].index(BSDF) # Ordering important, must match the order of the fwd/bwdPbrBSDF kernel.
    return _optix_env_shade_func.apply(optix_ctx, mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, iBSDF, n_samples_x, rnd_seed, shadow_scale)

def bilateral_denoiser(col, nrm, zdz, sigma):
    col_w = _bilateral_denoiser_func.apply(col, nrm, zdz, sigma)
    return col_w[..., 0:3] / col_w[..., 3:4]
