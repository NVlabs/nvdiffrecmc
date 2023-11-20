# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Build main image using dockerfile
docker build -f Dockerfile -t $1 .

# Install tiny-cuda-nn into the image file using runtime. 
# This is a workaround to avoid having to reconfigure docker to build images using nvidia runtime: https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime.
# Changing docker settings require root privileges.
rm ./tmp.cid
docker run --gpus device=0 --cidfile tmp.cid -it $1 pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
docker commit $(< tmp.cid) $1
rm ./tmp.cid