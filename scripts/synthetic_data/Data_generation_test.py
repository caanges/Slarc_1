import bpy
import random
import math
import json
import mathutils
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import os
import time

# Get objects
scene = bpy.context.scene
colection_0 = bpy.data.collections["Collection"]
colection_1 = bpy.data.collections["Collection2"]

camera = bpy.data.objects['Camera']
ugv = bpy.data.objects['UGV']
wheel11 = bpy.data.objects['wheel11']
wheel12 = bpy.data.objects['wheel12']
wheel21 = bpy.data.objects['wheel21']
wheel22 = bpy.data.objects['wheel22']
SUN = bpy.data.objects['Sun']

# Output folder
output_path = r"H:\Programmering\dva513\Slarc_1\Data\Data_img\Gen_data"
labeling_path = r"H:\Programmering\dva513\Slarc_1\Data\Data_label\json_data"

dataset = {}  

img_width = scene.render.resolution_x
img_height = scene.render.resolution_y

# Add light if not exists
if 'Light' not in bpy.data.objects:
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))

    light = bpy.data.objects['Light']
    light.data.energy = 5

def visibility(obj, camera,scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.location
    target = obj.location
    direction = (target - origin).normalized()

    hit, loc, normal, index, hit_obj, matrix = scene.ray_cast(
            depsgraph, origin, direction
            )
    if hit:
        return True
    return False

def get_bbox(obj, camera, scene):

    coords = []

    # get 8 corners of bounding box
    for corner in obj.bound_box:
        world_coord = obj.matrix_world @ mathutils.Vector(corner)

        co_2d = world_to_camera_view(scene, camera, world_coord)

        x = co_2d.x * img_width
        y = (1 - co_2d.y) * img_height

        coords.append((x, y))

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    return xmin, xmax, ymin, ymax

def make_json_data(obj, camera, scene, pic_num):
    xmin, xmax, ymin, ymax = get_bbox(obj, camera, scene)

    # convert to YOLO format
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
   
    if visibility(obj, camera, scene):
        dataset[pic_num] = {
        "object": obj.name,
        "bbox": [x_center, y_center, w, h],
        "camera_location": list(camera.location),
        "camera_rotation": list(camera.rotation_euler)
        }
    else:
        dataset[pic_num] = {
        "object": "ignore",
        "bbox": [x_center, y_center, w, h],
        "camera_location": list(camera.location),
        "camera_rotation": list(camera.rotation_euler)
        }
       
def save_json():
    global labeling_path, dataset
    json_path = os.path.join(labeling_path, "dataset.json")

    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=4)

def Generate_data():
    global ugv, wheel11, wheel12, wheel21, wheel22, SUN, camera, scene

    radius = 5
    radius_1 = 40
    loop_size_r = 5
    loop_size_a = 1
    camera.location.z = 60
    camera.location.x = 0
    camera.location.y = 0
    num_of_loops = 0
    num_loop_one = 0
    camera.rotation_euler[0] = 0
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0

    for i in range(loop_size_a):
        direction = ugv.location + Vector((0, 10, 0)) - camera.location
        direction_1 = ugv.location + Vector((0, 10, 0)) - SUN.location
        camera.rotation_euler = direction.to_track_quat('-Z').to_euler()
        SUN.rotation_euler = direction_1.to_track_quat('-Z', 'Y').to_euler()
        camera.rotation_euler[0] = 0
        camera.rotation_euler[1] = 0
        #camera.location.z -= 1
    
        for i in range(loop_size_r):
            angle = i * 0.1
            SUN.location.x = radius_1 * math.cos(angle)
            SUN.location.y = radius_1 * math.sin(angle)
            camera.location.x = radius * math.cos(angle)
            camera.location.y = radius * math.sin(angle)
            camera.rotation_euler[2] = angle
            # render image
            img_path = os.path.join(output_path, f"img_{num_loop_one:04d}.png")
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)
       
            make_json_data(ugv, camera, scene, num_of_loops)
            make_json_data(wheel11, camera, scene, num_of_loops + 1)
            make_json_data(wheel12, camera, scene, num_of_loops + 2)
            make_json_data(wheel21, camera, scene, num_of_loops + 3)
            make_json_data(wheel22, camera, scene, num_of_loops + 4)
       
            num_of_loops += 5
            num_loop_one += 1

def main():
    Generate_data()
    print("Data Generated!")

main()