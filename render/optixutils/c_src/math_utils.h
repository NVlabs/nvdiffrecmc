// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once 

#ifdef __CUDACC__

template<class T> static __device__ __inline__ T clamp(T x, T _min, T _max) { return min(_max, max(_min, x)); }
static __device__ inline float3 make_float3(float a) { return make_float3(a, a, a); }

static __device__ inline float2&   operator/=  (float2& a, const float2& b)       { a.x /= b.x; a.y /= b.y; return a; }
static __device__ inline float2&   operator*=  (float2& a, const float2& b)       { a.x *= b.x; a.y *= b.y; return a; }
static __device__ inline float2&   operator+=  (float2& a, const float2& b)       { a.x += b.x; a.y += b.y; return a; }
static __device__ inline float2&   operator-=  (float2& a, const float2& b)       { a.x -= b.x; a.y -= b.y; return a; }
static __device__ inline float2&   operator/=  (float2& a, float b)               { a.x /= b; a.y /= b; return a; }
static __device__ inline float2&   operator*=  (float2& a, float b)               { a.x *= b; a.y *= b; return a; }
static __device__ inline float2&   operator+=  (float2& a, float b)               { a.x += b; a.y += b; return a; }
static __device__ inline float2&   operator-=  (float2& a, float b)               { a.x -= b; a.y -= b; return a; }
static __device__ inline float2    operator/   (const float2& a, const float2& b) { return make_float2(a.x / b.x, a.y / b.y); }
static __device__ inline float2    operator*   (const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
static __device__ inline float2    operator+   (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
static __device__ inline float2    operator-   (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
static __device__ inline float2    operator/   (const float2& a, float b)         { return make_float2(a.x / b, a.y / b); }
static __device__ inline float2    operator*   (const float2& a, float b)         { return make_float2(a.x * b, a.y * b); }
static __device__ inline float2    operator+   (const float2& a, float b)         { return make_float2(a.x + b, a.y + b); }
static __device__ inline float2    operator-   (const float2& a, float b)         { return make_float2(a.x - b, a.y - b); }
static __device__ inline float2    operator/   (float a, const float2& b)         { return make_float2(a / b.x, a / b.y); }
static __device__ inline float2    operator*   (float a, const float2& b)         { return make_float2(a * b.x, a * b.y); }
static __device__ inline float2    operator+   (float a, const float2& b)         { return make_float2(a + b.x, a + b.y); }
static __device__ inline float2    operator-   (float a, const float2& b)         { return make_float2(a - b.x, a - b.y); }
static __device__ inline float2    operator-   (const float2& a)                  { return make_float2(-a.x, -a.y); }
static __device__ inline float3&   operator/=  (float3& a, const float3& b)       { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
static __device__ inline float3&   operator*=  (float3& a, const float3& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ inline float3&   operator+=  (float3& a, const float3& b)       { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ inline float3&   operator-=  (float3& a, const float3& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ inline float3&   operator/=  (float3& a, float b)               { a.x /= b; a.y /= b; a.z /= b; return a; }
static __device__ inline float3&   operator*=  (float3& a, float b)               { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ inline float3&   operator+=  (float3& a, float b)               { a.x += b; a.y += b; a.z += b; return a; }
static __device__ inline float3&   operator-=  (float3& a, float b)               { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ inline float3    operator/   (const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
static __device__ inline float3    operator*   (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ inline float3    operator+   (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ inline float3    operator-   (const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ inline float3    operator/   (const float3& a, float b)         { return make_float3(a.x / b, a.y / b, a.z / b); }
static __device__ inline float3    operator*   (const float3& a, float b)         { return make_float3(a.x * b, a.y * b, a.z * b); }
static __device__ inline float3    operator+   (const float3& a, float b)         { return make_float3(a.x + b, a.y + b, a.z + b); }
static __device__ inline float3    operator-   (const float3& a, float b)         { return make_float3(a.x - b, a.y - b, a.z - b); }
static __device__ inline float3    operator/   (float a, const float3& b)         { return make_float3(a / b.x, a / b.y, a / b.z); }
static __device__ inline float3    operator*   (float a, const float3& b)         { return make_float3(a * b.x, a * b.y, a * b.z); }
static __device__ inline float3    operator+   (float a, const float3& b)         { return make_float3(a + b.x, a + b.y, a + b.z); }
static __device__ inline float3    operator-   (float a, const float3& b)         { return make_float3(a - b.x, a - b.y, a - b.z); }
static __device__ inline float3    operator-   (const float3& a)                  { return make_float3(-a.x, -a.y, -a.z); }
static __device__ inline float4&   operator/=  (float4& a, const float4& b)       { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }
static __device__ inline float4&   operator*=  (float4& a, const float4& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ inline float4&   operator+=  (float4& a, const float4& b)       { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ inline float4&   operator-=  (float4& a, const float4& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ inline float4&   operator/=  (float4& a, float b)               { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }
static __device__ inline float4&   operator*=  (float4& a, float b)               { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ inline float4&   operator+=  (float4& a, float b)               { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ inline float4&   operator-=  (float4& a, float b)               { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ inline float4    operator/   (const float4& a, const float4& b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
static __device__ inline float4    operator*   (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ inline float4    operator+   (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ inline float4    operator-   (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ inline float4    operator/   (const float4& a, float b)         { return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); }
static __device__ inline float4    operator*   (const float4& a, float b)         { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ inline float4    operator+   (const float4& a, float b)         { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ inline float4    operator-   (const float4& a, float b)         { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ inline float4    operator/   (float a, const float4& b)         { return make_float4(a / b.x, a / b.y, a / b.z, a / b.w); }
static __device__ inline float4    operator*   (float a, const float4& b)         { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ inline float4    operator+   (float a, const float4& b)         { return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ inline float4    operator-   (float a, const float4& b)         { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __device__ inline float4    operator-   (const float4& a)                  { return make_float4(-a.x, -a.y, -a.z, -a.w); }

static __device__ inline float sum(float3 a)
{
    return a.x + a.y + a.z;
}

static __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static __device__ inline void bwd_dot(float3 a, float3 b, float3& d_a, float3& d_b, float d_out)
{
    d_a.x += d_out * b.x; d_a.y += d_out * b.y; d_a.z += d_out * b.z;
    d_b.x += d_out * a.x; d_b.y += d_out * a.y; d_b.z += d_out * a.z;
}

static __device__ inline float luminance(const float3 rgb)
{
    return dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
}

static __device__ inline float3 cross(float3 a, float3 b)
{
    float3 out;
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - a.x * b.z;
    out.z = a.x * b.y - a.y * b.x;
    return out;
}

static __device__ inline void bwd_cross(float3 a, float3 b, float3 &d_a, float3 &d_b, float3 d_out)
{
    d_a.x += d_out.z * b.y - d_out.y * b.z;
    d_a.y += d_out.x * b.z - d_out.z * b.x;
    d_a.z += d_out.y * b.x - d_out.x * b.y;

    d_b.x += d_out.y * a.z - d_out.z * a.y;
    d_b.y += d_out.z * a.x - d_out.x * a.z;
    d_b.z += d_out.x * a.y - d_out.y * a.x;
}

static __device__ inline float3 reflect(float3 x, float3 n)
{
    return n * 2.0f * dot(n, x) - x;
}

static __device__ inline void bwd_reflect(float3 x, float3 n, float3& d_x, float3& d_n, float3 d_out)
{
    d_x.x += d_out.x * (2 * n.x * n.x - 1) + d_out.y * (2 * n.x * n.y) + d_out.z * (2 * n.x * n.z);
    d_x.y += d_out.x * (2 * n.x * n.y) + d_out.y * (2 * n.y * n.y - 1) + d_out.z * (2 * n.y * n.z);
    d_x.z += d_out.x * (2 * n.x * n.z) + d_out.y * (2 * n.y * n.z) + d_out.z * (2 * n.z * n.z - 1);

    d_n.x += d_out.x * (2 * (2 * n.x * x.x + n.y * x.y + n.z * x.z)) + d_out.y * (2 * n.y * x.x) + d_out.z * (2 * n.z * x.x);
    d_n.y += d_out.x * (2 * n.x * x.y) + d_out.y * (2 * (n.x * x.x + 2 * n.y * x.y + n.z * x.z)) + d_out.z * (2 * n.z * x.y);
    d_n.z += d_out.x * (2 * n.x * x.z) + d_out.y * (2 * n.y * x.z) + d_out.z * (2 * (n.x * x.x + n.y * x.y + 2 * n.z * x.z));
}

static __device__ inline float3 safe_normalize(float3 v)
{
    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return l > 0.0f ? (v / l) : make_float3(0.0f);
}

static __device__ inline void bwd_safe_normalize(const float3 v, float3& d_v, float3 d_out)
{

    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (l > 0.0f)
    {
        float fac = 1.0 / powf(v.x * v.x + v.y * v.y + v.z * v.z, 1.5f);
        d_v.x += (d_out.x * (v.y * v.y + v.z * v.z) - d_out.y * (v.x * v.y) - d_out.z * (v.x * v.z)) * fac;
        d_v.y += (d_out.y * (v.x * v.x + v.z * v.z) - d_out.x * (v.y * v.x) - d_out.z * (v.y * v.z)) * fac;
        d_v.z += (d_out.z * (v.x * v.x + v.y * v.y) - d_out.x * (v.z * v.x) - d_out.y * (v.z * v.y)) * fac;
    }
}

// Code from 
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
static __device__ inline void branchlessONB(const float3 &n, float3 &b1, float3 &b2)
{
    float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = make_float3(b, sign + n.y * n.y * a, -n.y);
}

#endif