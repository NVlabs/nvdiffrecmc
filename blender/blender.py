# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# Simple script to show how to load our assets in Blender
# 
# Open Blender 3.x, click the scripting tab and open this script
# Run the script (Alt P)
# Under the shading tab, you will see the shading network and environent probe (World node)
# You can then render the model using the Cycles renderer
import os
import bpy 
import numpy as np

# path to your mesh
MESH_PATH = "../out/bob/mesh"

RESOLUTION = 512
SAMPLES = 64

################### Renderer settings ###################
bpy.ops.file.pack_all()
scene = bpy.context.scene
scene.world.use_nodes = True
scene.render.engine = 'CYCLES'
scene.render.film_transparent = True
scene.cycles.device = 'GPU'
scene.cycles.samples = SAMPLES
scene.cycles.max_bounces = 0
scene.cycles.diffuse_bounces = 0
scene.cycles.glossy_bounces = 0
scene.cycles.transmission_bounces = 0
scene.cycles.volume_bounces = 0
scene.cycles.transparent_max_bounces = 8
scene.cycles.use_denoising = True
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

################### Image output ###################

# PNG output with sRGB tonemapping
scene.display_settings.display_device = 'sRGB'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

# OpenEXR output, no tonemapping applied
# scene.display_settings.display_device = 'None'
# scene.view_settings.view_transform = 'Standard'
# scene.view_settings.exposure = 0.0
# scene.view_settings.gamma = 1.0
# scene.render.image_settings.file_format = 'OPEN_EXR'


################### Import obj mesh ###################

imported_object = bpy.ops.import_scene.obj(filepath=os.path.join(MESH_PATH, "mesh.obj"), axis_forward = '-Z', axis_up = 'Y')
obj_object = bpy.context.selected_objects[0]

################### Fix material graph ###################

# Get material node tree, find BSDF and specular texture
material = obj_object.active_material
bsdf = material.node_tree.nodes["Principled BSDF"]
image_node_ks = bsdf.inputs["Specular"].links[0].from_node

# Split the specular texture into metalness and roughness
separate_node = material.node_tree.nodes.new(type="ShaderNodeSeparateRGB")
separate_node.name="SeparateKs"
material.node_tree.links.new(image_node_ks.outputs[0], separate_node.inputs[0])
material.node_tree.links.new(separate_node.outputs[2], bsdf.inputs["Metallic"])
material.node_tree.links.new(separate_node.outputs[1], bsdf.inputs["Roughness"])

normal_map_node = bsdf.inputs["Normal"].links[0].from_node
texture_n_node = normal_map_node.inputs["Color"].links[0].from_node
material.node_tree.links.remove(normal_map_node.inputs["Color"].links[0])
normal_separate_node = material.node_tree.nodes.new(type="ShaderNodeSeparateRGB")
normal_separate_node.name="SeparateNormal"
normal_combine_node = material.node_tree.nodes.new(type="ShaderNodeCombineRGB")
normal_combine_node.name="CombineNormal"

normal_invert_node = material.node_tree.nodes.new(type="ShaderNodeMath")
normal_invert_node.name="InvertNormal"
normal_invert_node.operation='SUBTRACT'
normal_invert_node.inputs[0].default_value = 1.0

material.node_tree.links.new(texture_n_node.outputs[0], normal_separate_node.inputs['Image'])
material.node_tree.links.new(normal_separate_node.outputs['R'], normal_combine_node.inputs['R'])
material.node_tree.links.new(normal_separate_node.outputs['G'], normal_invert_node.inputs[1])
material.node_tree.links.new(normal_invert_node.outputs[0], normal_combine_node.inputs['G'])
material.node_tree.links.new(normal_separate_node.outputs['B'], normal_combine_node.inputs['B'])
material.node_tree.links.new(normal_combine_node.outputs[0], normal_map_node.inputs["Color"])

material.node_tree.links.remove(bsdf.inputs["Specular"].links[0])

# Set default values
bsdf.inputs["Specular"].default_value = 0.5
bsdf.inputs["Specular Tint"].default_value = 0.0
bsdf.inputs["Sheen Tint"].default_value = 0.0
bsdf.inputs["Clearcoat Roughness"].default_value = 0.0

################### Load HDR probe ###################

texcoord = scene.world.node_tree.nodes.new(type="ShaderNodeTexCoord")
mapping = scene.world.node_tree.nodes.new(type="ShaderNodeMapping")
mapping.inputs['Rotation'].default_value = [0, 0, -np.pi*0.5]
envmap = scene.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
envmap.image = bpy.data.images.load(os.path.join(MESH_PATH, "probe.hdr"))

scene.world.node_tree.links.new(envmap.outputs['Color'], scene.world.node_tree.nodes['Background'].inputs['Color'])
scene.world.node_tree.links.new(texcoord.outputs['Generated'], mapping.inputs['Vector'])
scene.world.node_tree.links.new(mapping.outputs['Vector'], envmap.inputs['Vector'])