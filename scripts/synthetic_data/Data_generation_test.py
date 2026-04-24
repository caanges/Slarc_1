import bpy
import random
import math
import json
import mathutils
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import os
import time
import math

# Get objects
scene = bpy.context.scene

# Output folder
output_path = r"H:\Programmering\dva513\Slarc_1\Data\Data_img\Gen_data"
labeling_path = r"H:\Programmering\dva513\Slarc_1\Data\Data_label\json_data"

dataset = {}  

img_width = scene.render.resolution_x
img_height = scene.render.resolution_y

def get_bbox(obj, camera, scene):

    coords = []

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

def change_of_scene(collection_id):
    collection_0 = bpy.data.collections["Collection"]
    collection_1 = bpy.data.collections["Collection2"]
    collection_2 = bpy.data.collections["Collection3"]
    collection_3 = bpy.data.collections["Collection4"]

    if collection_id == 1:
        collection_0.hide_render = False
        collection_1.hide_render = True
        collection_2.hide_render = True
        collection_3.hide_render = True
    elif collection_id == 2:
        collection_0.hide_render = True
        collection_1.hide_render = False
        collection_2.hide_render = True
        collection_3.hide_render = True
    elif collection_id == 3:
        collection_0.hide_render = True
        collection_1.hide_render = True
        collection_2.hide_render = False
        collection_3.hide_render = True
    elif collection_id == 4:
        collection_0.hide_render = True
        collection_1.hide_render = True
        collection_2.hide_render = True
        collection_3.hide_render = False

def make_json_data(obj, camera, scene):
    xmin, xmax, ymin, ymax = get_bbox(obj, camera, scene)

    #match the new data place with the place of the existing data
    global_id = len(dataset)

    # convert to YOLO format
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
   
    #save data for each object 
    dataset[global_id] = {
    "object": obj.name,
    "bbox": [x_center, y_center, w, h],
    "camera_location": list(camera.location),
    "camera_rotation": list(camera.rotation_euler)
    }
       
def save_json():
    global labeling_path, dataset
    json_path = os.path.join(labeling_path, "dataset.json")

    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=4)

def change_sun(SUN):
    strength = random.uniform(0.1, 20)
    SUN.data.energy = strength
    angle = random.uniform(0, 10)
    SUN.data.angle = math.pi * cos(angle)

def Generate_data(num, ugv, wheel11, wheel12, wheel21, wheel22, SUN, camera, scene):
    #initialize all the parameters
    radius = 2
    radius_1 = 20
    loop_size_r = 5
    loop_size_a = 1
    camera.location.z = 20
    camera.location.x = 0
    camera.location.y = 0
    num_of_loops = 0
    num_loop_one = 0
    camera.rotation_euler[0] = 0
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0

    for i in range(loop_size_a):
        #first loop
        direction = ugv.location + Vector((0, 10, 0)) - camera.location
        direction_1 = ugv.location + Vector((0, 10, 0)) - SUN.location
        camera.rotation_euler = (0, 0, 0)  
        SUN.rotation_euler = direction_1.to_track_quat('-Z', 'Y').to_euler()
        camera.rotation_euler[0] = 0
        camera.rotation_euler[1] = 0
    
        for i in range(loop_size_r):
            #change the sun on a presett intervall Ex. 1:10
            if (i % 10) == 0:
                change_sun(SUN)

            #change the angle and rotation of the camera around the 'UGV'
            angle = i * 0.5
            camera.location.x = radius * math.cos(angle)
            camera.location.y = radius * math.sin(angle)
            camera.rotation_euler[2] = angle

            # render image and make a path with the neccesary data in the name of each picture
            img_path = os.path.join(output_path, f"img{num}_{num_loop_one:04d}.png")
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)
       
            #make the Json data for each object
            make_json_data(ugv, camera, scene)
            make_json_data(wheel11, camera, scene)
            make_json_data(wheel12, camera, scene)
            make_json_data(wheel21, camera, scene)
            make_json_data(wheel22, camera, scene)

            #increase num_of_loops by five to match the amount of parameters we are making data for 
            num_of_loops += 5
            num_loop_one += 1

def main():
    #set the number of scenes existing
    number_off_scenes = 4
    for i in range(0, number_off_scenes):
        #change the object to be meassured each time the scene changes
        camera = bpy.data.objects[f'Camera.{i:03d}']
        ugv = bpy.data.objects[f'UGV.{i:03d}']
        wheel11 = bpy.data.objects[f'wheel11.{i:03d}']
        wheel12 = bpy.data.objects[f'wheel12.{i:03d}']
        wheel21 = bpy.data.objects[f'wheel21.{i:03d}']
        wheel22 = bpy.data.objects[f'wheel22.{i:03d}']
        SUN = bpy.data.objects[f'Sun.{i:03d}']
        #change the camera so it does not keep the data from the previos data
        scene.camera = camera
        change_of_scene(i + 1)
        Generate_data(i, ugv, wheel11, wheel12, wheel21, wheel22, SUN, camera, scene)
    #save the json file when done with generating all the picture and correctly appending the Json data
    save_json()
    print("Data Generated!")

main()
