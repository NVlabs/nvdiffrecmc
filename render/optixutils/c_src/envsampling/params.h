// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../accessor.h"

struct EnvSamplingParams
{
    // Ray data
    PackedTensorAccessor32<float, 4>    ro;             // ray origin
    
    // GBuffer
    PackedTensorAccessor32<float, 3>    mask;
    PackedTensorAccessor32<float, 4>    gb_pos;
    PackedTensorAccessor32<float, 4>    gb_pos_grad;
    PackedTensorAccessor32<float, 4>    gb_normal;
    PackedTensorAccessor32<float, 4>    gb_normal_grad;
    PackedTensorAccessor32<float, 4>    gb_view_pos;
    PackedTensorAccessor32<float, 4>    gb_kd;
    PackedTensorAccessor32<float, 4>    gb_kd_grad;
    PackedTensorAccessor32<float, 4>    gb_ks;
    PackedTensorAccessor32<float, 4>    gb_ks_grad;
    
    // Light
    PackedTensorAccessor32<float, 3>    light;
    PackedTensorAccessor32<float, 3>    light_grad;
    PackedTensorAccessor32<float, 2>    pdf;        // light pdf
    PackedTensorAccessor32<float, 1>    rows;       // light sampling cdf
    PackedTensorAccessor32<float, 2>    cols;       // light sampling cdf

    // Output
    PackedTensorAccessor32<float, 4>    diff;
    PackedTensorAccessor32<float, 4>    diff_grad;
    PackedTensorAccessor32<float, 4>    spec;
    PackedTensorAccessor32<float, 4>    spec_grad;

    // Table with random permutations for stratified sampling
    PackedTensorAccessor32<int, 2>      perms;

    OptixTraversableHandle              handle;
    unsigned int                        BSDF;
    unsigned int                        n_samples_x;
    unsigned int                        rnd_seed;
    unsigned int                        backward;
    float                               shadow_scale;
};

