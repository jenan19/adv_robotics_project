import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import shutil

PATH_TO_DIR = os.getcwd() 
PATH_TO_HULL_COLORED = PATH_TO_DIR+ "/ply_no_concave"
PATH_TEST = PATH_TO_DIR+ "/hull_colored_test"
PATH_TRAIN = PATH_TO_DIR+ "/hull_colored_train"
PATH_PCD = PATH_TO_DIR+ "/pcd_no_concave"

if os.path.exists(PATH_TO_HULL_COLORED) is False:
    os.mkdir(PATH_TO_HULL_COLORED)

files = os.listdir(PATH_PCD)
for pcd_ in files:
    pcd = o3d.io.read_point_cloud(PATH_PCD + "/"+ pcd_)
    name = pcd_[:-3] + "ply"
    o3d.io.write_point_cloud(PATH_TO_HULL_COLORED + "/" + name, pcd)



# if os.path.exists(PATH_TEST) is False:
#     os.mkdir(PATH_TEST)
# if os.path.exists(PATH_TRAIN) is False:
#     os.mkdir(PATH_TRAIN)
# files = os.listdir(PATH_TO_HULL_COLORED)
# files = np.array(files)
# np.random.shuffle(files)
# n = len(files)
# n_train = (n*0.10)
# for idx, file_ in enumerate(files):
#     if idx < n - n_train:
#         shutil.copyfile(PATH_TO_HULL_COLORED + "/"+file_,PATH_TRAIN + "/"+file_ )
#     else:
#         shutil.copyfile(PATH_TO_HULL_COLORED + "/"+file_,PATH_TEST + "/"+file_ )


