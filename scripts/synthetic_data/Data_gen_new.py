import bpy
import random
import math
import mathutils
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import os
import time

#initialize scene
scene = bpy.context.scene

output_path = r"C:\Data_dva513\Data\Train_val_test"
object_data = []
class_map = {}
collections = []
random_obj_collections = []
random_obj = 7

img_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
img_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

theta_start = 0
theta_end = 2 * math.pi
last_loop = 0
phi_start = 0
phi_end = math.radians(25)
radius = 0.5

#_______Collections_________#
def init_scenes(num_of_scenes):
    #Initialize every scene by entering them into the collections list
    for i in range(0, num_of_scenes):
        collections.append(bpy.data.collections[f'Collection.{i:02d}']) 

def init_random_obj(random_obj):
    #initialize every random object by enter them into the object collections list
    for i in range(1, random_obj + 1):
        random_obj_collections.append(bpy.data.collections[f'Rand_coll.{i:02d}'])

def disable_collection(collection, state):
    #disable or enable the collections based on the desired state for the collectino
    for obj in collection.all_objects:
        obj.hide_set(state)
    collection.hide_render = state

def change_of_scene(collection_id, num_of_scenes):
    #get the state to be used in the disable_collection() function
    for i in range(0, num_of_scenes):
        if i == collection_id:
            state = False
        else:
            state = True
        disable_collection(collections[i], state)

def handle_object(collection_id, state):
    #disable or enable diferento objects based on randomised value of object in object list
    for obj in random_obj_collections[collection_id].all_objects:
        obj.hide_set(state)
    random_obj_collections[collection_id].hide_render = state
    
#________Blur_________#
def setup_compositor(use_blur=False, blur_strength = 8):
    scene = bpy.context.scene
    scene.use_nodes = True

    #create nodes to allow for blur in image after render
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    #clear the nodes so old data is not stored
    nodes.clear()

    render_layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")

    if use_blur:
        blur = nodes.new("CompositorNodeBlur")
        blur.size_x = blur_strength
        blur.size_y = blur_strength

        links.new(render_layers.outputs["Image"], blur.inputs["Image"])
        links.new(blur.outputs["Image"], composite.inputs["Image"])
    else:
        links.new(render_layers.outputs["Image"], composite.inputs["Image"])

#__________data_handling_________#
def map_objects(obj):
    name = obj.name.split('.')[0]
    name_ugv = obj.name.split('.')[0]
    if name.startswith("KEYPOINT_"):
        KEYPOINT_id = int(name.split("_")[1])
        class_map[name] = KEYPOINT_id
    else:
        ugv_id = 15
        class_map[name_ugv] = ugv_id

def get_bbox(obj, camera, scene):
    coords = []

    #put every keypoints x, y, w, h in the list coords[]
    #keypoints use only x and y coordinates wich is calculated later and the width and height are discared. 
    #Bounding box is used for the UGV
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
    #check if the keypoints is visible in order to get a visibility score needed for YOLO formating 
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    target = obj.matrix_world @ Vector((0, 0, 0))
    origin = camera.matrix_world.translation

    direction = (target - origin).normalized()
    distance = (target - origin).length

    #change the origin slightly by moving it towards the camera such that it is not hiden by it's own features
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
    if not (0 <= co_2D.x <= 1 and 0 <= co_2D.y <= 1):
        return 0

    if hit and hit_obj is not None:
        if hit_obj.name.startswith("UGV") or hit_obj.name.startswith("KEYPOINT"):
            if hit_obj == obj:
                return 2
            if hit_obj != obj:
                return 0
    return 0

def save_yolo_format(text_path, bbox, keypoints):
    label_path = text_path
    class_id = 0
    cx, cy, w, h = bbox

    #save the the data in YOLO format by having all data in one line
    with open(label_path, "w") as f:
        line = f"{class_id} {cx} {cy} {w} {h} "
        line += " ".join(map(str, keypoints))
        f.write(line + "\n")

#___________Objects_____________#
def change_ugv_loc(ugv):
    #change the ugv location to get diferent angles 
    ugv_x = 0
    ugv_y = 0

    ugv.location.x = ugv_x
    ugv.location.y = ugv_y
    ugv.rotation_euler[2] = random.uniform(0, math.pi)

def camera_location(camera, ugv, loop_2, loop_1, orbit_radius, num_pic_H, num_pic_V): 
    #move the camera in a semi circle   
    global last_loop, radius
    
    phi = phi_start + (loop_1 / (num_pic_V - 1)) * (phi_end - phi_start)
    theta = theta_start + (loop_2 / num_pic_H) * 2 * math.pi 
    if last_loop != loop_1:
        z = camera.location.z - 0.1
        radius += +0.5
    else:
        z = camera.location.z 

    x = ugv.location.x + radius  * math.cos(theta)
    y = ugv.location.y + radius * math.sin(theta)
    camera.location = (x, y, z)
    direction = ugv.location - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    last_loop = loop_1
    bpy.context.view_layer.update()

def change_sun(Sun, ugv, loop_1, loop_2):
    #change the sun to get diferent angles on the sun and shadows with a varying strength of the sun
    radius = 20
    theta = loop_2 * 0.1
    alpha = loop_1 * 0.2

    x = ugv.location.x + radius * math.cos(theta)
    y = ugv.location.y + radius * math.sin(theta)
    z = ugv.location.z + 30 * abs(math.sin(alpha))

    Sun.data.energy = random.uniform(7.5, 12.5)
    sun_pos = mathutils.Vector((x, y, z))
    direction = ugv.location - sun_pos
    Sun.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.view_layer.update()

def change_random_obj():
    #get a random number of objects to be used and based on that choose that amount of objects and randomize what objects 
    #is choosen
    for i in range(0, len(random_obj_collections)):
        handle_object(i, False)

    random_obj_num = random.randint(0, len(random_obj_collections))
    chosen = random.sample(range(len(random_obj_collections)), random_obj_num)

    for idx in chosen:
        handle_object(idx, True)

#____________Data Generation_______________#
def Generate_data(num, ugv, key_points, Sun, camera, scene, num_attr, level, scene_min_lev):
    global radius
    loop_size_i = 10
    loop_size_j = 5
    loop_counter_1 = 0
    loop_counter_2 = 0
    random.seed(num)
    
    #__________Initialize the camera_______#
    camera.location.x = ugv.location.x + random.uniform(-0.2, 0.2)
    camera.location.y = ugv.location.x + random.uniform(-0.2, 0.2)
    camera.location.z = 30 - (level * 5) 
    #for every new time Generate_data is run the camera goes down by (level * 5) and then every new scene level is set to 
    #zero making it start back at Z location 30 and then go down num_level times, geting closer photos of the UGV
    camera.rotation_euler[0] = 0
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0
    orbit_radius = 0.5
    radius = 0.5
    #initialize the sun
    change_sun(Sun, ugv, 1, 1)

    for i in range(loop_size_i):
        change_ugv_loc(ugv)
        
        change_random_obj()
        for j in range(loop_size_j):
            bpy.context.view_layer.update()
            object_data.clear()
            
            camera_location(camera, ugv, loop_counter_2, loop_counter_1, orbit_radius, loop_size_j, loop_size_i)
            change_sun(Sun, ugv, loop_counter_1 + 1, loop_counter_2 + 1)
            cx, cy, w, h = get_bbox(ugv, camera, scene)

            key_point_list = []
            for k in range(num_attr):
                kp = key_points[k]
                x, y, disc_w, disc_h = get_bbox(kp, camera, scene)

                vis = visibility(kp, camera, scene)
                key_point_list.extend([x,y,vis])
        
            #get the validation data
            if num >= (6*5):
                output_path_img = os.path.join(output_path, r"images\val")
                img_path = os.path.join(output_path_img, f"img{num}_{loop_counter_2:04d}.png")

                output_path_txt = os.path.join(output_path, r"labels\val") 
                txt_path = os.path.join(output_path_txt, f"img{num}_{loop_counter_2:04d}.txt")
            #get training data
            else:
                output_path_img = os.path.join(output_path, r"images\train")
                img_path = os.path.join(output_path_img, f"img{num}_{loop_counter_2:04d}.png")

                output_path_txt = os.path.join(output_path, r"labels\train")
                txt_path = os.path.join(output_path_txt, f"img{num}_{loop_counter_2:04d}.txt")
            
            blur_level = random.randint(3, 9)
            use_blur = (loop_counter_2 % 5 == 0)
            setup_compositor(use_blur=use_blur, blur_strength = blur_level)

            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)

            bbox = [cx, cy, w, h]
            save_yolo_format(txt_path, bbox, key_point_list)
            loop_counter_2 += 1
            bpy.context.view_layer.update()
        loop_counter_1 += 1

#_____main_____#
def main():
    num_scenes = 8
    levels = 5
    level = 0
    tot_dif_num = levels * num_scenes
    num_attr = 13
    key_points = {}
    scene_change = 0

    init_scenes(num_scenes) 
    init_random_obj(random_obj)
    change_of_scene(0, num_scenes)


    for i in range(tot_dif_num):
        scene_change = i // levels
        level = i % levels
        change_of_scene(scene_change, num_scenes)
        camera = bpy.data.objects[f'Camera.{scene_change:03d}']
        ugv = bpy.data.objects[f'UGV.{scene_change:03d}']

        for j in range(0, num_attr):
            key_points[j] = bpy.data.objects[f'KEYPOINT_{j}.{scene_change:03d}']
            map_objects(key_points[j])

        Sun = bpy.data.objects[f'Sun.{scene_change:03d}']
        scene.camera = camera
        bpy.context.view_layer.update()

        Generate_data(i, ugv, key_points, Sun, camera, scene, num_attr, level, scene_change)
        level += 1

    print("Data Generated!")
     
#____Run____#
main()
