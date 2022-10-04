// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#ifdef __CUDACC__

#define SPECULAR_EPSILON 1e-4f
#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

//------------------------------------------------------------------------
// Lambert functions

__device__ inline float fwdLambert(const float3 nrm, const float3 wi)
{
    return max(dot(nrm, wi) / M_PI, 0.0f);
}

__device__ inline void bwdLambert(const float3 nrm, const float3 wi, float3& d_nrm, float3& d_wi, const float d_out)
{
    if (dot(nrm, wi) > 0.0f)
        bwd_dot(nrm, wi, d_nrm, d_wi, d_out / M_PI);
}

//------------------------------------------------------------------------
// Fresnel Schlick 

__device__ inline float fwdFresnelSchlick(const float f0, const float f90, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = powf(1.0f - _cosTheta, 5.0f);
    return f0 * (1.0f - scale) + f90 * scale;
}

__device__ inline void bwdFresnelSchlick(const float f0, const float f90, const float cosTheta, float& d_f0, float& d_f90, float& d_cosTheta, const float d_out)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = pow(max(1.0f - _cosTheta, 0.0f), 5.0f);
    d_f0 += d_out * (1.0 - scale);
    d_f90 += d_out * scale;
    if (cosTheta >= SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
    {
        d_cosTheta += d_out * (f90 - f0) * -5.0f * powf(1.0f - cosTheta, 4.0f);
    }
}

__device__ inline float3 fwdFresnelSchlick(const float3 f0, const float3 f90, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = powf(1.0f - _cosTheta, 5.0f);
    return f0 * (1.0f - scale) + f90 * scale;
}

__device__ inline void bwdFresnelSchlick(const float3 f0, const float3 f90, const float cosTheta, float3& d_f0, float3& d_f90, float& d_cosTheta, const float3 d_out)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = pow(max(1.0f - _cosTheta, 0.0f), 5.0f);
    d_f0 += d_out * (1.0 - scale);
    d_f90 += d_out * scale;
    if (cosTheta >= SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
    {
        d_cosTheta += sum(d_out * (f90 - f0) * -5.0f * powf(1.0f - cosTheta, 4.0f));
    }
}

//------------------------------------------------------------------------
// Ndf GGX

__device__ inline float fwdNdfGGX(const float alphaSqr, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * M_PI);
}

__device__ inline void bwdNdfGGX(const float alphaSqr, const float cosTheta, float& d_alphaSqr, float& d_cosTheta, const float d_out)
{
    // Torch only back propagates if clamp doesn't trigger
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    d_alphaSqr += d_out * (1.0f - (alphaSqr + 1.0f) * cosThetaSqr) / (M_PI * powf((alphaSqr - 1.0) * cosThetaSqr + 1.0f, 3.0f));
    if (cosTheta > SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
    {
        d_cosTheta += d_out * -(4.0f * (alphaSqr - 1.0f) * alphaSqr * cosTheta) / (M_PI * powf((alphaSqr - 1.0) * cosThetaSqr + 1.0f, 3.0f));
    }
}

//------------------------------------------------------------------------
// Lambda GGX

__device__ inline float fwdLambdaGGX(const float alphaSqr, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    float res = 0.5f * (sqrtf(1.0f + alphaSqr * tanThetaSqr) - 1.0f);
    return res;
}

__device__ inline void bwdLambdaGGX(const float alphaSqr, const float cosTheta, float& d_alphaSqr, float& d_cosTheta, const float d_out)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    float res = 0.5f * (sqrtf(1.0f + alphaSqr * tanThetaSqr) - 1.0f);

    d_alphaSqr += d_out * (0.25 * tanThetaSqr) / sqrtf(alphaSqr * tanThetaSqr + 1.0f);
    if (cosTheta > SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
        d_cosTheta += d_out * -(0.5 * alphaSqr) / (powf(_cosTheta, 3.0f) * sqrtf(alphaSqr / cosThetaSqr - alphaSqr + 1.0f));
}

//------------------------------------------------------------------------
// Masking GGX

__device__ inline float fwdMaskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO)
{
    float lambdaI = fwdLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = fwdLambdaGGX(alphaSqr, cosThetaO);
    return 1.0f / (1.0f + lambdaI + lambdaO);
}

__device__ inline void bwdMaskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO, float& d_alphaSqr, float& d_cosThetaI, float& d_cosThetaO, const float d_out)
{
    // FWD eval
    float lambdaI = fwdLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = fwdLambdaGGX(alphaSqr, cosThetaO);

    // BWD eval
    float d_lambdaIO = -d_out / powf(1.0f + lambdaI + lambdaO, 2.0f);
    bwdLambdaGGX(alphaSqr, cosThetaI, d_alphaSqr, d_cosThetaI, d_lambdaIO);
    bwdLambdaGGX(alphaSqr, cosThetaO, d_alphaSqr, d_cosThetaO, d_lambdaIO);
}

//------------------------------------------------------------------------
// GGX specular

__device__ float3 fwdPbrSpecular(const float3 col, const float3 nrm, const float3 wo, const float3 wi, const float alpha, const float min_roughness)
{
    float _alpha = clamp(alpha, min_roughness * min_roughness, 1.0f);
    float alphaSqr = _alpha * _alpha;

    float3 h = safe_normalize(wo + wi);
    float woDotN = dot(wo, nrm);
    float wiDotN = dot(wi, nrm);
    float woDotH = dot(wo, h);
    float nDotH = dot(nrm, h);

    float D = fwdNdfGGX(alphaSqr, nDotH);
    float G = fwdMaskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN);
    float3 F = fwdFresnelSchlick(col, make_float3(1.0f), woDotH);
    float3 w = F * D * G * 0.25 / woDotN;

    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);
    return frontfacing ? w : make_float3(0.0f);
}

__device__ void bwdPbrSpecular(
    const float3 col, const float3 nrm, const float3 wo, const float3 wi, const float alpha, const float min_roughness,
    float3& d_col, float3& d_nrm, float3& d_wo, float3& d_wi, float& d_alpha, const float3 d_out)
{
    ///////////////////////////////////////////////////////////////////////
    // FWD eval

    float _alpha = clamp(alpha, min_roughness * min_roughness, 1.0f);
    float alphaSqr = _alpha * _alpha;

    float3 h = safe_normalize(wo + wi);
    float woDotN = dot(wo, nrm);
    float wiDotN = dot(wi, nrm);
    float woDotH = dot(wo, h);
    float nDotH = dot(nrm, h);

    float D = fwdNdfGGX(alphaSqr, nDotH);
    float G = fwdMaskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN);
    float3 F = fwdFresnelSchlick(col, make_float3(1.0f), woDotH);
    float3 w = F * D * G * 0.25 / woDotN;
    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);

    if (frontfacing)
    {
        ///////////////////////////////////////////////////////////////////////
        // BWD eval

        float3 d_F = d_out * D * G * 0.25f / woDotN;
        float d_D = sum(d_out * F * G * 0.25f / woDotN);
        float d_G = sum(d_out * F * D * 0.25f / woDotN);

        float d_woDotN = -sum(d_out * F * D * G * 0.25f / (woDotN * woDotN));

        float3 d_f90 = make_float3(0);
        float d_woDotH = 0, d_wiDotN = 0, d_nDotH = 0, d_alphaSqr = 0;
        bwdFresnelSchlick(col, make_float3(1.0f), woDotH, d_col, d_f90, d_woDotH, d_F);
        bwdMaskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN, d_alphaSqr, d_woDotN, d_wiDotN, d_G);
        bwdNdfGGX(alphaSqr, nDotH, d_alphaSqr, d_nDotH, d_D);

        float3 d_h = make_float3(0);
        bwd_dot(nrm, h, d_nrm, d_h, d_nDotH);
        bwd_dot(wo, h, d_wo, d_h, d_woDotH);
        bwd_dot(wi, nrm, d_wi, d_nrm, d_wiDotN);
        bwd_dot(wo, nrm, d_wo, d_nrm, d_woDotN);

        float3 d_h_unnorm = make_float3(0);
        bwd_safe_normalize(wo + wi, d_h_unnorm, d_h);
        d_wo += d_h_unnorm;
        d_wi += d_h_unnorm;

        if (alpha > min_roughness * min_roughness)
            d_alpha += d_alphaSqr * 2 * alpha;
    }
}

//------------------------------------------------------------------------
// Full PBR BSDF

__device__ void fwdPbrBSDF(const float3 kd, const float3 arm, const float3 pos, const float3 nrm, const float3 view_pos, const float3 wi, const float min_roughness, float3 &diffuse, float3 &specular)
{
    float3 wo = safe_normalize(view_pos - pos);

    float alpha = arm.y * arm.y;
    float3 spec_col = (make_float3(0.04f) * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x);
    // Removed because of demodulated albedo.
    // float3 diff_col = kd * (1.0f - arm.z);

    float diff = 0.0f;
    diff = fwdLambert(nrm, wi);
    
    diffuse = make_float3(diff);//diff_col * diff;
    specular = fwdPbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness);
}

__device__ void bwdPbrBSDF(
    const float3 kd, const float3 arm, const float3 pos, const float3 nrm, const float3 view_pos, const float3 wi, const float min_roughness,
    float3& d_kd, float3& d_arm, float3& d_pos, float3& d_nrm, float3& d_view_pos, float3& d_wi, const float3 d_diffuse, float3 d_specular)
{
    ////////////////////////////////////////////////////////////////////////
    // FWD
    float3 _wo = view_pos - pos;
    float3 wo = safe_normalize(_wo);

    float alpha = arm.y * arm.y;
    float3 spec_col = (make_float3(0.04f) * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x);

    ////////////////////////////////////////////////////////////////////////
    // BWD

    float d_alpha = 0;
    d_wi = make_float3(0);
    float3 d_spec_col = make_float3(0), d_wo = make_float3(0);
    bwdPbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness, d_spec_col, d_nrm, d_wo, d_wi, d_alpha, d_specular);

    // float d_diff = sum(diff_col * d_diffuse);
    float d_diff = sum(d_diffuse);
    bwdLambert(nrm, wi, d_nrm, d_wi, d_diff);

    // Backprop: spec_col = (0.04f * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x)
    d_kd -= d_spec_col * (arm.x - 1.0f) * arm.z;
    d_arm.x += sum(d_spec_col * (arm.z * (make_float3(0.04f) - kd) - 0.04f));
    d_arm.z -= sum(d_spec_col * (kd - make_float3(0.04f)) * (arm.x - 1.0f));

    // Backprop: alpha = arm.y * arm.y
    d_arm.y += d_alpha * 2 * arm.y;

    // Backprop: float3 wo = safe_normalize(view_pos - pos);
    float3 d__wo = make_float3(0);
    bwd_safe_normalize(_wo, d__wo, d_wo);
    d_view_pos += d__wo;
    d_pos -= d__wo;
}

#endif