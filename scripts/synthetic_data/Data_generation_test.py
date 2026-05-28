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

scene.use_nodes = True
tree = scene.node_tree
links = tree.links

# Clear old nodes
tree.nodes.clear()

render_layers = tree.nodes.new(type='CompositorNodeRLayers')
blur_node = tree.nodes.new(type='CompositorNodeBlur')
composite_node = tree.nodes.new(type='CompositorNodeComposite')

# Blur settings
blur_node.filter_type = 'GAUSS'
blur_node.size_x = 6
blur_node.size_y = 6

# Output folder
output_path = r"C:\Data_dva513\Data\Train_val_test"

object_data = []
class_map = {}

img_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
img_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

def disable_collection(collection, state):
    for obj in collection.all_objects:
        obj.hide_set(state)     
    collection.hide_render = state    

def visiblity(obj, camera, scene):
    scene.view_layers.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    target = obj.matrix_world @ Vector((0, 0, 0))
    origin = camera.matrix_world.translation

    direction = (target - origin).normalized()
    distance = (target - origin).length

    origin = origin + direction * 0.01

    hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
        depsgraph,
        origin,
        direction,
        distance = distance
    )

    co_2d = world_to_camera_view(scene, camera, target)

    if co_2d.z <= 0:
        return 0
    if not (0 <= co_2d.x <= 1 and 0 <= co_2d.y <= 1):
        return 0

    if hit and hit_obj is not None:
        if hit_obj.name.startswith("UGV") or hit_obj.name.startswith("KEYPOINT"):
            # allow occlusion only if it's NOT the target chain
            if hit_obj == obj:
                return 2
            if hit_obj != obj:
                return 0
    
    return 0

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
    collection_5 = bpy.data.collections["Collection6"]
    collection_6 = bpy.data.collections["Collection7"]
    collection_7 = bpy.data.collections["Collection8"]

    if collection_id == 1:
        disable_collection(collection_0, False)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
        disable_collection(collection_5, True)
        disable_collection(collection_6, True)
        disable_collection(collection_7, True)
    elif collection_id == 2:
        disable_collection(collection_0, True)
        disable_collection(collection_1, False)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
        disable_collection(collection_5, True)
        disable_collection(collection_6, True)
        disable_collection(collection_7, True)
    elif collection_id == 3:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, False)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
        disable_collection(collection_5, True)
        disable_collection(collection_6, True)
        disable_collection(collection_7, True)
    elif collection_id == 4:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, False)
        disable_collection(collection_4, True)
        disable_collection(collection_5, True)
        disable_collection(collection_6, True)
        disable_collection(collection_7, True)
    elif collection_id == 5:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, False)
        disable_collection(collection_5, True)
        disable_collection(collection_6, True)
        disable_collection(collection_7, True)
    elif collection_id == 6:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
        disable_collection(collection_5, False)
        disable_collection(collection_6, True)
        disable_collection(collection_7, True)
    elif collection_id == 7:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
        disable_collection(collection_5, True)
        disable_collection(collection_6, False)
        disable_collection(collection_7, True)
    elif collection_id == 8:
        disable_collection(collection_0, True)
        disable_collection(collection_1, True)
        disable_collection(collection_2, True)
        disable_collection(collection_3, True)
        disable_collection(collection_4, True)
        disable_collection(collection_5, True)
        disable_collection(collection_6, True)
        disable_collection(collection_7, False)

def save_yolo_form(text_path, bbox, keypoints):
    label_path = text_path
    
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

def change_sun(SUN, target, i):
    #strength = random.uniform(0.5, 10)
    SUN.data.energy = 5
    
    radius = 10
    angle = i * 0.5

    x = target.location.x + radius * math.cos(angle)
    y = target.location.y + radius * math.sin(angle)
    z = target.location.z + 10 * abs(math.sin(angle)) 

    sun_pos = mathutils.Vector((x, y, z))
    direction = target.location - sun_pos
    SUN.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.view_layer.update()


def Generate_data(num, ugv, key_points, SUN, camera, scene):
    #initialize all the parameters
    radius = 2
    loop_size_r = 25
    loop_size_a = 10
    num_attr = 13
    camera.location.z = 20
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


        loc_x = ugv.location.x 
        loc_y = ugv.location.y

        ugv.location.x = loc_x + random.uniform(-1, 1)
        ugv.location.y = loc_y + random.uniform(-1, 1) 

        change_sun(SUN, ugv, i)

        camera.location.x = ugv.location.x + random.uniform(-1, 1)
        camera.location.y = ugv.location.y + random.uniform(-1, 1)
        camera.location.z = random.uniform(5, 25)

        for j in range(loop_size_r):
            scene.view_layers.update()
            object_data.clear()

            #change the angle and rotation of the camera around the 'UGV'
            angle = j * 0.1
            camera.rotation_euler[2] = angle
            orbit_radius = random.uniform(0.5, 2.0)

            camera.location.x = (
                ugv.location.x
                + orbit_radius * math.cos(angle)
                + random.uniform(-0.3, 0.3)
            )

            camera.location.y = (
                ugv.location.y
                + orbit_radius * math.sin(angle)
                + random.uniform(-0.3, 0.3)
            )

            bpy.context.view_layer.update()

            xmin, xmax, ymin, ymax = get_bbox(ugv, camera, scene)

            cx = (xmin + xmax) / 2 / img_width
            cy = (ymin + ymax) / 2 / img_height
            w  = (xmax - xmin) / img_width
            h  = (ymax - ymin) / img_height

            key_point_list = []
            for k in range(13):
                kp = key_points[k]

                xmin, xmax, ymin, ymax = get_bbox(kp, camera, scene)

                x = (xmin + xmax) / 2 / img_width
                y = (ymin + ymax) / 2 / img_height

                vis = visiblity(kp, camera, scene)

                key_point_list.extend([x,y,vis])

            if num == 6 or num == 7:
                output_path_img = os.path.join(output_path, r"images\val")
                img_path = os.path.join(output_path_img, f"img{num}_{num_loop_one:04d}.png")

                output_path_txt = os.path.join(output_path, r"labels\val")
                text_path = os.path.join(output_path_txt, f"img{num}_{num_loop_one:04d}.txt")

            else:
                output_img = os.path.join(output_path, r"images\train")
                img_path = os.path.join(output_img, f"img{num}_{num_loop_one:04d}.png")
            
                output_text = os.path.join(output_path, r"labels\train")
                text_path = os.path.join(output_text, f"img{num}_{num_loop_one:04d}.txt")


            # Every 25th image becomes blurry
            if num_loop_one % 25 == 0:

                # Random blur amount
                blur_strength = random.randint(5, 15)

                blur_node.size_x = blur_strength
                blur_node.size_y = blur_strength

                # Connect render -> blur -> output
                links.clear()

                links.new(
                    render_layers.outputs['Image'],
                    blur_node.inputs['Image']
                )

                links.new(
                    blur_node.outputs['Image'],
                    composite_node.inputs['Image']
                )

                scene.render.filepath = img_path
                bpy.ops.render.render(write_still=True)

            else:

                # No blur
                links.clear()

                links.new(
                    render_layers.outputs['Image'],
                    composite_node.inputs['Image']
                )

                scene.render.filepath = img_path
                bpy.ops.render.render(write_still=True)


            for kp in key_points.values():
                object_data_app(kp, camera, scene)

            bbox = [cx, cy, w, h]
            
            save_yolo_form(text_path, bbox, key_point_list)
            num_loop_one += 1
            scene.view_layers.update()

def main():
    #set the number of scenes existing
    number_off_scenes = 8
    num_attr = 13
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
