// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "denoising.h"

#define FLT_EPS 0.0001f

__global__ void bilateral_denoiser_fwd_kernel(BilateralDenoiserParams params)
{
    uint3 idx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);

    if (idx.z >= params.col.size(0) || idx.y >= params.col.size(1) || idx.x >= params.col.size(2))
        return;

    // Fetch central tap
    float3 c_nrm = fetch3(params.nrm, idx.z, idx.y, idx.x);
    float2 c_zdz = fetch2(params.zdz, idx.z, idx.y, idx.x);

    float variance = params.sigma * params.sigma;
    int filter_rad = 2 * ceil(params.sigma * 2.5) + 1;

    float accum_w = 0.0f;
    float3 accum_col = make_float3(0.0f);
    for (int32_t fy = -filter_rad; fy <= filter_rad; ++fy)
    {
        for (int32_t fx = -filter_rad; fx <= filter_rad; ++fx)
        {
            // Compute tap coordinates, used for input activations and bilateral guides
            int32_t y = idx.y + fy;
            int32_t x = idx.x + fx;

            if (y < 0 || x < 0 || y >= params.col.size(1) || x >= params.col.size(2))
                continue;

            // Fetch current tap
            float3 t_col = fetch3(params.col, idx.z, y, x);
            float3 t_nrm = fetch3(params.nrm, idx.z, y, x);
            float2 t_zdz = fetch2(params.zdz, idx.z, y, x);

            /////////////////////////////////////////////////////////
            // Compute bilateral weight
            /////////////////////////////////////////////////////////

            // Distance
            float dist_sqr = fx * fx + fy * fy;
            float dist = sqrtf(dist_sqr);
            float w_xy = expf(-dist_sqr / (2.0f * variance));

            // Normal
            float w_normal = powf(min(max(dot(t_nrm, c_nrm), FLT_EPS), 1.0f), 128.0f);

            // Depth
            float w_depth = expf(-(abs(t_zdz.x - c_zdz.x) / max(c_zdz.y * dist, FLT_EPS)));

            float w = w_xy * w_normal * w_depth;

            accum_col = accum_col + t_col * w;
            accum_w += w;
        }
    }

    params.out[idx.z][idx.y][idx.x][0] = accum_col.x;
    params.out[idx.z][idx.y][idx.x][1] = accum_col.y;
    params.out[idx.z][idx.y][idx.x][2] = accum_col.z;
    params.out[idx.z][idx.y][idx.x][3] = max(accum_w, 0.0001f);
}

__global__ void bilateral_denoiser_bwd_kernel(BilateralDenoiserParams params)
{
    uint3 idx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);

    if (idx.z >= params.col.size(0) || idx.y >= params.col.size(1) || idx.x >= params.col.size(2))
        return;

    // Fetch central tap
    float3 c_nrm = fetch3(params.nrm, idx.z, idx.y, idx.x);
    float2 c_zdz = fetch2(params.zdz, idx.z, idx.y, idx.x);

    float variance = params.sigma * params.sigma;
    int filter_rad = 2 * ceil(params.sigma * 2.5) + 1;

    float3 accum_grad = make_float3(0.0f);
    for (int32_t fy = -filter_rad; fy <= filter_rad; ++fy)
    {
        for (int32_t fx = -filter_rad; fx <= filter_rad; ++fx)
        {
            // Compute tap coordinates, used for input activations and bilateral guides
            int32_t y = idx.y + fy;
            int32_t x = idx.x + fx;

            if (y < 0 || x < 0 || y >= params.col.size(1) || x >= params.col.size(2))
                continue;

            // Fetch current tap
            float3 t_col = fetch3(params.col, idx.z, y, x);
            float3 t_nrm = fetch3(params.nrm, idx.z, y, x);
            float2 t_zdz = fetch2(params.zdz, idx.z, y, x);

            /////////////////////////////////////////////////////////
            // Compute bilateral weight
            /////////////////////////////////////////////////////////

            // Distance, transposing fx & fy doesn't affect distance
            float dist_sqr = fx * fx + fy * fy;
            float dist = sqrtf(dist_sqr);
            float w_xy = expf(-dist_sqr / (2.0f * variance));

            // Normal, transpose c_ and t_ (it's symmetric so doesn't matter)
            float w_normal = powf(min(max(dot(t_nrm, c_nrm), FLT_EPS), 1.0f), 128.0f);

            // Depth, transpose c_ and t_ (matters for the denominator)
            float w_depth = expf(-(abs(t_zdz.x - c_zdz.x) / max(t_zdz.y * dist, FLT_EPS)));

            float w = w_xy * w_normal * w_depth;

            float3 t_col_grad = w * fetch3(params.out_grad, idx.z, y, x);
            accum_grad += t_col_grad;
        }
    }

    params.col_grad[idx.z][idx.y][idx.x][0] = accum_grad.x;
    params.col_grad[idx.z][idx.y][idx.x][1] = accum_grad.y;
    params.col_grad[idx.z][idx.y][idx.x][2] = accum_grad.z;
}
