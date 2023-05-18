import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy import stats
from tqdm import tqdm

PATH_TO_DIR = os.getcwd() 
PATH_TO_TRANSLATION = PATH_TO_DIR + "/translation/"
PATH_TO_CSV = PATH_TO_DIR + "/csv_distances/"
METER_TO_MM = 1000.0
PATH_PCDFILE = PATH_TO_DIR + "/pcd/"
PATH_CADFILE = PATH_TO_DIR + "/out/"
GREY = [0.5, 0.5, 0.5]
PATH_TO_HULL_COLORED = PATH_TO_DIR+ "/hull_colored_ds"


def read_translation_txt_file(file_):
    f = open( file_, "r")
    translation = np.array([float(i) for i in f.read().split(' ')])
    f.close()
    return translation*METER_TO_MM

def load_point_cloud(file_):
    pcd = o3d.io.read_point_cloud(PATH_PCDFILE + file_.split(".")[0]+ ".pcd")
    
    # Reverse transformation in measurement
    translation = read_translation_txt_file(PATH_TO_TRANSLATION + file_.split(".")[0]+ ".txt")
    translation[1] = -translation[1]
    Rx = R.from_euler('x', 180, degrees=True).as_matrix()
    
    # Scale from mm to m 
    pcd.scale(METER_TO_MM,[0,0,0])
    
    # Transform points
    pcd = pcd.translate(translation)
    pcd = pcd.rotate(Rx, [0,0,0])
    
    # Color points gray
    pcd.paint_uniform_color(GREY)
    
    return pcd
    

def convert_mesh_to_pcd(file_, number_of_points):
    mesh = o3d.io.read_triangle_mesh(PATH_CADFILE + file_)
    
    
    pcd = mesh.sample_points_poisson_disk(number_of_points, init_factor=5, pcl=None)
    #pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    return pcd

def MADThresholding(obs, thresh = 3):
    if len(obs.shape) == 1:
        obs = obs[:,None]
    median = np.median(obs, axis=0)
    diff = np.sum((obs - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh




def color_pcd_hull(pcd_hull, pcd_cad, file_):
    
    xyz_hull =np.asarray(pcd_hull.points)
    xyz_cad =np.asarray(pcd_cad.points)
    
    kdtree=KDTree(xyz_cad)
    dist,points = kdtree.query(xyz_hull,1, workers=-1) 

    mu = np.mean(dist)
    median = np.median(dist)
    #sigma = 2*np.std(dist)
    
    
    mask = MADThresholding(dist)
    #print(mask)
    

    with open(PATH_TO_CSV + file_.split(".")[0]+ ".csv", "w") as f:
        f.write("point, dist\n")
        
        for i , point in enumerate(points):

            f.write("%f,%f \n" % (point,dist[i]))
            # DOES NOT WORK AS INTENDED
            if pcd_cad.colors[point][0] > 0.5:
                pcd_hull.colors[i][0]= 1

            if mask[i]:
                pcd_hull.colors[i][1]  = 1

        f.write("%f \n" % mu)
        f.write("%f \n" % median)         
         

    return pcd_hull


def main():
    
    if os.path.exists(PATH_TO_HULL_COLORED) is False:
        os.mkdir(PATH_TO_HULL_COLORED)

    # for idx, file_ in enumerate(os.listdir(PATH_PLYFILE)):
    for file_ in tqdm(os.listdir(PATH_CADFILE)):
        if len(file_.split('.')) == 2:
            pcd_hull = load_point_cloud(file_)
            pcd_cad = convert_mesh_to_pcd(file_,10000)
            #o3d.visualization.draw_geometries([pcd_cad])
            pcd_hull = color_pcd_hull(pcd_hull, pcd_cad, file_)
            o3d.io.write_point_cloud(PATH_TO_HULL_COLORED +"/"+ file_, pcd_hull)
        

if __name__ == "__main__":
    main()


