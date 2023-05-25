from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

simulation_app.set_setting('persistent/app/viewport/displayOptions', 31983)

import os
from PIL import Image
from omni.isaac.core import SimulationContext
from scipy.spatial.transform import Rotation as R
import time
import omni.isaac.core.utils.extensions as extensions_utils
from pxr import Gf, UsdPhysics, Usd, UsdGeom
import omni.usd
from omni.physx import get_physx_interface
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import delete_prim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils import transformations as tf
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.utils import rotations as r
from omni.isaac.core.objects import DynamicCuboid
import matplotlib.pyplot as plt

from omni.isaac.manipulators.grippers import ParallelGripper

from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import carb
import sys
from omni.isaac.core.utils.bounds import compute_aabb

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    # Use carb to log warnings, errors and infos in your application (shown on terminal)
    carb.log_error("Could not find nucleus server with /Isaac folder")


my_world = World(stage_units_in_meters=1.0)
my_world.set_simulation_dt(rendering_dt=1/30)

add_reference_to_stage(usd_path="/home/tk/Downloads/light.usd",prim_path="/World/light")
add_reference_to_stage(usd_path="/home/tk/Downloads/cube.usd",prim_path="/World/cube")
add_reference_to_stage(usd_path="/home/tk/Downloads/cube.usd",prim_path="/World/cube2")
my_world.scene.add(XFormPrim(prim_path="/World/kiwi", name="kiwi"))
cube = my_world.scene.add(XFormPrim(prim_path="/World/cube", name="cube",position=[0.3, 0, 0.0],orientation=[0,0,0,1]))
cube2 = my_world.scene.add(XFormPrim(prim_path="/World/cube2", name="cube2",position=[0.3, 0, 0.0],orientation=[0,0,0,1]))
light = my_world.scene.add(XFormPrim(prim_path="/World/light", name="light"))

cam = Camera(prim_path="/World/camera", name="camera", position=[0.3, 0, 0.0],orientation=r.euler_angles_to_quat([0,0,180],degrees=True),frequency=10, resolution=(2592, 2048))


my_world.scene.add(XFormPrim("/World/point", name="point"))

cam.set_focal_length(2.0701013565)
cam.set_vertical_aperture(1.6336017987)
cam.set_clipping_range(0.001)

file_path = os.path.join(os.getcwd(), "visual_hull", "")
dir_path = os.path.dirname(file_path) + "/"


path_in = "/home/tk/Downloads/cad_model_colored/usdfiles/"
i = 0

my_world.reset()
cube = my_world.scene.get_object("cube")
cam.initialize()
np.set_printoptions(suppress = True)


def rot_z_matrix(rot, gamma):
    cy = np.cos(gamma)
    sy = np.sin(gamma)
    rot_z = np.asarray([[cy, -sy, 0], 
                        [sy, cy, 0], 
                        [0, 0, 1]])
    return np.matmul(rot_z,rot)

def rot_y_matrix(rot, gamma):
    cy = np.cos(gamma)
    sy = np.sin(gamma)
    rot_z = np.asarray([[cy, 0, sy], 
                        [0, 1, 0], 
                        [sy, 0, cy]])
    return np.matmul(rot_z,rot)

def rot_y_transform(rot, gamma):
    cy = np.cos(gamma)
    sy = np.sin(gamma)
    rot_y = np.asarray([[cy, 0, sy], 
                        [0, 1, 0], 
                        [-sy, 0, cy]])
    return np.matmul(rot, rot_y) 



i = 0
dist_cam = cam.get_world_pose()[0]
orien_cam = cam.get_world_pose()[1]
cam_intrinic = np.asarray(cam.get_intrinsics_matrix())

def rotate_object(my_world, current_object, cam):
    count = 0
    last_frame_number = -1
    cam_pos = []
    image_frames = []
    while(True):
        my_world.step(render=True)
        current_frame = cam.get_current_frame()
        frame_number = current_frame['rendering_frame']
        if frame_number > last_frame_number:
            current_orientation = np.asarray(r.quat_to_rot_matrix(current_object.get_world_pose()[1]))
            image_frames.append(current_frame['rgba'])
            res = np.asarray(rot_z_matrix(np.asarray(current_orientation), np.deg2rad(10)))

            pos = np.matmul(res, np.asarray(dist_cam))

            desired_orientation = r.euler_angles_to_quat(r.matrix_to_euler_angles(res))
            res2 = np.asarray(rot_z_matrix(np.asarray(current_orientation), np.deg2rad(180)))
            cube_pos = cube.get_world_pose()
            # cam_pos.append({'translation':cube_pos[0], 'orientation':r.quat_to_rot_matrix(cube_pos[1])})

            t = cube_pos[0]
            R = r.quat_to_rot_matrix(cube_pos[1])
            # t_res = rot_y_transform(t,np.deg2rad(90))
            R_res = rot_y_transform(res2,np.deg2rad(90))
            cam_pos.append({'translation':cube_pos[0], 'orientation':R_res})

            cube.set_world_pose(position=pos, orientation=r.euler_angles_to_quat(  r.matrix_to_euler_angles(res2) ))
            cube2.set_world_pose(position=pos, orientation=r.euler_angles_to_quat(  r.matrix_to_euler_angles(R_res) ))
            current_object.set_world_pose(orientation=desired_orientation)
            count += 1
            last_frame_number = frame_number
            if count == 37:
                return cam_pos, image_frames
                break

path_in = "/home/tk/Downloads/cad_model_colored/usdfiles/"

files_usd = os.listdir(path_in)

def load_object(file, idx):
    name  =  str(file.split('.')[0])
    prim_name = name.replace("-","_")
    prim_path=str("/World/n_" + str(idx))
    add_reference_to_stage(usd_path=path_in + file,prim_path=str("/World/n_" + str(idx)))
    prim = get_prim_at_path(str("/World/n_" + str(idx)))
    inner_path = str(prim.GetChildren()).split("<")[1].split(">")[0]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    box  = compute_aabb(bbox_cache=bbox_cache,prim_path=prim_path, include_children=True)
    # print(box)
    box = box*0.001
    x_min =box[0]
    x_max =box[3]
    y_min =box[1]
    y_max =box[4]
    x = -((box[0] + box[3])/2)
    y = -((box[1] + box[4])/2)
    XFormPrim(prim_path=inner_path, name="n_" + str(idx), scale=[0.001, 0.001, 0.001], position=[x,y,0])
    component = XFormPrim(prim_path=prim_path)
    z_min = box[2] 
    box[0] = x_min + x
    box[1] = x_max + x
    box[2] = y_min + y
    box[3] = y_max + y 
    box[4] = -box[5] 
    box[5] = -z_min 
    return component, prim_path, box

def write_image(file, cam_pos, image_frames, bbox):
    root_path =  "/home/tk/Downloads/cad_model_colored/images/"
    file_name  =  str(file.split('.')[0])
    current_folder = os.path.join(root_path, file_name)
    os.mkdir(current_folder)
    current_folder += "/"
    for k in range(0, len(image_frames)-1):
        rgb_img = Image.fromarray(image_frames[k], "RGBA")
        rgb_img.save(current_folder + file_name +  "_"+str(k+1) +".png")

    with open(current_folder + file_name +'_par.txt', 'w') as f:
        f.write(' '.join(map(str, np.around(bbox,decimals=10)))+ '\n')
        for k in range(0, len(cam_pos)-1):
            f.write(str(k + 1)+ ' ' +' '.join(map(str,cam_intrinic.flatten())) +' '+ 
                    ' '.join(map(str,np.around(cam_pos[k]['orientation'].flatten(),decimals=5)))+' '+ 
                    ' '.join(map(str,np.around(cam_pos[k]['translation'].flatten(), decimals=5)))+'\n')

current = 0


while simulation_app.is_running():
    my_world.step(render=True)
    if i >= 100:
        for idx,file in enumerate( os.listdir(path_in)):
            current, prim_path, bbox = load_object(file, idx)
            for j in range(10):
                my_world.step(render=True)
            cam_pos, image_frames = rotate_object(my_world,current, cam)
            write_image(file, cam_pos, image_frames,bbox)
            delete_prim(prim_path) 
            my_world.reset()

    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        pass
    i += 1
simulation_app.close()
