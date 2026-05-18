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

#_______Collections_________#
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

#__________data_handling_________#
def map_objects(obj):
    name = obj.name.split('.')[0]
    name_ugv = obj.name.split('.')[0]
    if name.startswith("KEYPOINT_"):
        KEYPOINT_id = int(name.split("_")[1])
        class_map[name] = KEYPOINT_id
    else:
        ugv_id = 15
        class_map[name_ugv] = UGV_id

def get_bbox(obj, camera, scene):
    coords = []

    for corner in obj.bound_box:
        world_coord = obj.matrix_world @ mathutils.Vector(corner)
        co_2D = world_to_camera_view(scene, camera, world_coord)

        x = co_2D.x * img_width
        y = (1 - co_2D.y) * img_height
        coords.append((x,y))

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    x = (xmin + xmax) / 2 / img_width
    y = (ymin + ymax) / 2 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
    
    return x, y, w, h

def visibility(obj, camera, scene):
    scene.view_layers,update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    target = obj.matrix_world @ Vector((0, 0, 0))
    origin = camera.matrix_world.translateion

    direction = (target - origin).normalized()
    distance = (target - origin).length

    origin = origin + direction * 0.01

    hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance = distance
        )
    co_2D = world_to_camera_view(scene, camera, target)
    if co_2D.z <= 0:
        return 0
    if not (0 <= co_2D.x <= 1 and 0 <= co2D.y <= 1):
        return 0

    if hit and hit_obj is not None:
        if hot_obj.name.startswith("UGV") or hit_obj.name.startswith("KEYPOINT"):
            if hit_obj = obj:
                return 2
            if hit_obj != obj:
                return 0
    return 0

def object_data_app(obj, camera, scene):
    x, y, w, h = get_bbox(obj, camera, scene)

    check = visibility(obj, camera, scene)
    object_data.append({
        "object": obj.name,
        "bbox": [x_center, y_center, w, h],
        "visibility": check
    })

def save_yolo_format(text_path, bbox, key_point_list):
    label_path = text_path
    class_id = 0
    cx, cy, w, h = bbox

    with open(label_path, "w") as f:
        line = f"{class_id} {cx} {cy} {w} {h} "
        line += " ".join(map(str, keypoints))
        f.write(line + "\n")

#___________Objects_____________#
def change_ugv_loc(ugv):
    ugv_x = ugv.location.x
    ugv_y = ugv.location.y

    ugv.location.x = ugv_x + random.uniform(-1, 1)
    ugv.location.y = ugv_y + random.uniform(-1, 1)

def camera_location(camera, ugv, loop_2):

    orbit_radius = random.uniform(0.5, 2.0)
    angle = loop_2 * 0.1
    
    camera.location.x = (
            ugv.location.x
            + orbit_radius * math.cos(angle)
            + random.uniform(-0.3, 0.3)
        )
    camera.location.y = (
            ugv.location.y
            + orbit_radius * math.cos(angle)
            +random.uniform(-0.3, 0.3)
        )

    camera.location.z -= loop_2
    bpy.context.view_layer.update()


def change_sun(Sun, ugv, loop_1):
    radius = 20
    angle = i * 0.1

    x = ugv.location.x + radius * math.cos(angle)
    y = ugv.location.y + radius * math.sin(angle)
    z = target.location.z + 10 * abs(math.sin(angle))

    sun_pos = mathutils.Vector((x, y, z))
    direction = ugv.location - sun_pos
    Sun.rotatino_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.view_layer.update()

#____________Data Generation_______________#
def Generate_data(i, ugv, key_points, Sun, camera, scene, num_attr):
    loop_size_i = 15
    loop_size_j = 10

    camera.location.z = 20
    loop_counter_1 = 0
    loop_counter_2 = 0
    camera.rotation_euler[0] = 0
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0

    for i in range(loop_size_i):
        change_ugv_loc(ugv)
        change_sun(Sun, ugv, loop_counter_1)
        for j in range(loop_size_j):
            scene.view_layers.update()
            object_data.clear()
            
            camera_location(camera, ugv, loop_counter_2)

            cx, cy, w, h = get_bbox(ugv, camera, scene)

            key_point_list = []
            for k in range(num_attr):
                kp = keypoints[k]
                x, y, disc_w, disc_h = get_bbox(kp, camera, scene)

                vis = visibility(kp, camera, scene)
                key_point_list.extend([x,y,vis])
        
            if i == 6 or num == 7:
                output_path_img = os.path.join(output_path, r"images\val")
                img_path = os.path.join(output_path_img, f"img{i}_{loop_counter_1:04d}.png")

                output_path_txt = os.path.join(output_path, r"labels\val")
                txt_path = os.path.join(output_path_img, f"img{i}_{loop_counter_1:04d}.png")

            else:
                output_path_img = os.path.join(output_path, r"images\train")
                img_path = os.path.join(output_path_img, f"img{i}_{loop_counter_1:04d}.png")

                output_path_txt = os.path.join(output_path, r"labels\train")
                txt_path = os.path.join(output_path_img, f"img{i}_{loop_counter_1:04d}.png")
            
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)

            for kp in key_points.values():
                object_data_app(kp, camera, scene)

            bbox = [cx, cy, w, h]
            save_yolo_form(text_path, bbox, key_point_list)
            loop_counter_1 += 1
            scene.view_layers.update()

        
#_____main_____#
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

    print("Data Generated!")
     
#____Run____#
main()
