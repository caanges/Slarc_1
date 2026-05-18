import bpy
import random
import math
import mathutils
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import os
import time

scene = bpy.context.scene
output_path = r"C:\Data_dva513\Data\Train_val_test"
object_data = []
class_map = {}
collections = []

img_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
img_hwight = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

def init_scenes(num_of_scenes):
    for i in range(0, num_of_scenes):
        collections[i] = bpy.data.collections[f'Collection.{i:02d}']

def disable_collection(collection, state):
    for obj in collection.all_objects:
        obj.hide_set(state)
    collection.hide_render = state

def change_of_scene(collection_id, num_of_scenes):
    state = True
    for i in range(0, num_of_scenes):
        if i == collection_id:
            state = False
        disable_collection(collections[i], state)

def map_objects(obj):
    name = obj.name.split('.')[0]
    name_ugv = obj.name.split('.')[0]
    if name.startswith("KEYPOINT_"):
        KEYPOINT_id = int(name.split("_")[1])
        class_map[name] = KEYPOINT_id
    else:
        ugv_id = 15
        class_map[name_ugv] = UGV_id

def Generate_data(i, ugv, key_points, Sun, camera, scene, num_attr):
    loop_size_i = 25
    loop_size_j = 10

    camera.location.z = 20
    loop_counter = 0
    camera.rotation_euler[0] = 0
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0

    for i in range(loop_size_i):


def main():
    num_scenes = 2
    num_attr = 13
    key_points = {}

    init_scenes(num_scenes)

    for i in range(0, num_scenes):
        camera = bpy.data.objects[f'Camera.{i:03d}']
        ugv = bpy.data.objects[f'UGV.{i:03d}']
        for j in range(0, num_attr):
            key_points[j] = bpy.data.objects[f'KEYPOINT_{j}.{i:03d}']
            map_objects(key_points[j])

        sun = bpy.data.objects[f'Sun.{i:03d}']
        scene.camera = camera
        change_of_scene(i + 1, num_scenes)
        Generate_data(i, ugv, key_points, Sun, camera, scene, num_attr)

main()
