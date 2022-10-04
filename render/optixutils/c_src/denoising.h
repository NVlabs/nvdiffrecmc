// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "accessor.h"

struct BilateralDenoiserParams
{
    PackedTensorAccessor32<float, 4> col;
    PackedTensorAccessor32<float, 4> col_grad;  
    PackedTensorAccessor32<float, 4> nrm;
    PackedTensorAccessor32<float, 4> zdz;
    PackedTensorAccessor32<float, 4> out;
    PackedTensorAccessor32<float, 4> out_grad;
    float sigma;
};
