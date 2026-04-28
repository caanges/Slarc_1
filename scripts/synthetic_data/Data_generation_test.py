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

object_data = []
class_map = {}

img_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
img_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

def disable_collection(collection, state):
    for obj in collection.all_objects:
        obj.hide_set(state)     
    collection.hide_render = state    

def visiblity(obj, camera, scene):
    for corner in obj.bound_box:
        world_coord = obj.matrix_world @ Vector(corner)
        co_2d = world_to_camera_view(scene, camera, world_coord)

        if 0 <= co_2d.x <= 1 and 0 <= co_2d.y <= 1 and co_2d.z > 0:
            return 2  # visible
        elif co_2d.z > 0:
            return 1  # in front but outside frame
        else:
            return 0  # behind camera

def map_objects(obj):
    name = obj.name.split('.')[0]
    name_ugv = obj.name.split('.')[0]
    if name.startswith("KEYPOINT_"):
        KEYPOINT_id = int(name.split("_")[1])
        class_map[name] = KEYPOINT_id
    else:
        UGV_id = 15
        class_map[name_ugv] = UGV_id

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
    collection_4 = bpy.data.collections["Collection5"]

    if collection_id == 1:
        disable_collection(collection_0, False)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
    elif collection_id == 2:
        disable_collection(collection_0, True)
        disable_collection(collection_1, False)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
    elif collection_id == 3:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, False)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
    elif collection_id == 4:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, False)
        disable_collection(collection_4, True)
    elif collection_id == 5:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, False)

def save_yolo_form(img_path, bbox, keypoints):
    label_path = img_path.replace(".png", ".txt")
    
    class_id = 0
    cx, cy, w, h = bbox

    with open(label_path, "w") as f:
        line = f"{class_id} {cx} {cy} {w} {h} "
        line += " ".join(map(str, keypoints))
        f.write(line + "\n")

def object_data_app(obj, camera, scene):

    xmin, xmax, ymin, ymax = get_bbox(obj, camera, scene)

    # convert to YOLO format
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    check = visiblity(obj, camera, scene)

    object_data.append({
        "object": obj.name,
        "bbox": [x_center, y_center, w, h],
        "visibility": check
    })


def change_sun(SUN):
    strength = random.uniform(0.1, 20)
    SUN.data.energy = strength
    angle = random.uniform(0, 10)
    SUN.data.angle = math.pi * math.cos(angle)

def Generate_data(num, ugv, key_points, SUN, camera, scene):
    #initialize all the parameters
    radius = 2
    radius_1 = 20
    loop_size_r = 5
    loop_size_a = 1
    num_attr = 15
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
    
        for j in range(loop_size_r):
            object_data.clear()

            #change the sun on a presett intervall Ex. 1:10
            if (j % 10) == 0:
                change_sun(SUN)

            #change the angle and rotation of the camera around the 'UGV'
            angle = j * 0.5
            camera.location.x = radius * math.cos(angle)
            camera.location.y = radius * math.sin(angle)
            camera.rotation_euler[2] = angle

            bpy.context.view_layer.update()

            xmin, xmax, ymin, ymax = get_bbox(ugv, camera, scene)

            cx = (xmin + xmax) / 2 / img_width
            cy = (ymin + ymax) / 2 / img_height
            w  = (xmax - xmin) / img_width
            h  = (ymax - ymin) / img_height

            key_point_list = []
            for k in range(15):
                kp = key_points[k]

                xmin, xmax, ymin, ymax = get_bbox(kp, camera, scene)

                x = (xmin + xmax) / 2 / img_width
                y = (ymin + ymax) / 2 / img_height

                vis = visiblity(kp, camera, scene)

                key_point_list.extend([x,y,vis])

            # render image and make a path with the neccesary data in the name of each picture
            img_path = os.path.join(output_path, f"img{num}_{num_loop_one:04d}.png")
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)

            #increase num_of_loops by five to match the amount of parameters we are making data for 

            for kp in key_points.values():
                object_data_app(kp, camera, scene)

            bbox = [cx, cy, w, h]
            save_yolo_form(img_path, bbox, key_point_list)
            num_loop_one += 1

def main():
    #set the number of scenes existing
    number_off_scenes = 5
    num_attr = 15
    key_points = {}
    for i in range(0, number_off_scenes):
        #change the object to be meassured each time the scene changes
        camera = bpy.data.objects[f'Camera.{i:03d}']
        ugv = bpy.data.objects[f'UGV.{i:03d}']
        for j in range(0, num_attr):
            key_points[j] = bpy.data.objects[f'KEYPOINT_{j}.{i:03d}']
            map_objects(key_points[j])
        
        SUN = bpy.data.objects[f'Zun.{i:03d}']
        #change the camera so it does not keep the data from the previos data
        scene.camera = camera
        change_of_scene(i + 1)
        Generate_data(i, ugv, key_points, SUN, camera, scene)
    #save the json file when done with generating all the picture and correctly appending the Json data
    
    print("Data Generated!")

main()
