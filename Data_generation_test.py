import bpy
import random
import math
import json
import mathutils
from bpy_extras.object_utils import world_to_camera_view
import os 

# Get objects
scene = bpy.context.scene
camera = bpy.data.objects['Camera']
ugv = bpy.data.objects['UGV']

# Output folder (CHANGE THIS)
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

radius = 10
loop_size_r = 10
loop_size_a = 5
camera.location.z = 16
num_of_loops = 0
# Loop to generate images

for i in range(loop_size_a):
    num_of_loops += 1
    direction = ugv.location - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.location.z -= 1

    for i in range(loop_size_r):
        angle = i * 0.1

        loop_size_ra = i * num_of_loops

        camera.location.x = radius * math.cos(angle)
        camera.location.y = radius * math.sin(angle)
        # render image
        img_path = os.path.join(output_path, f"img_{num_of_loops:04d}.png")
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        # compute bbox
        xmin, xmax, ymin, ymax = get_bbox(ugv, camera, scene)

        # convert to YOLO format
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        dataset[img_path] = {
        "bbox": [x_center, y_center, w, h],
        "camera_location": list(camera.location),
        "camera_rotation": list(camera.rotation_euler)
        }
        num_of_loops += 1

json_path = os.path.join(labeling_path, "dataset.json")

with open(json_path, "w") as f:
    json.dump(dataset, f, indent=4)


print("Data Generated!")