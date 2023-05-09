import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
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

def color_pcd_hull(pcd_hull, pcd_cad, file_):
    
    xyz_hull =np.asarray(pcd_hull.points)
    xyz_cad =np.asarray(pcd_cad.points)
    
    kdtree=KDTree(xyz_cad)
    dist,points = kdtree.query(xyz_hull,1) 
    
    with open(PATH_TO_CSV + file_.split(".")[0]+ ".csv", "w") as f:
        f.write("point, dist\n")
        
        for i , point in enumerate(points):
            f.write("%s,%s/n" % (str(point),str(dist[i])))
            # band pass color value [0, 1] because open3D is dumb :)))))
            pcd_hull.colors[i] = [max(min(x, 1), 0) for x in pcd_cad.colors[point]]

    return pcd_hull


def main():
    
    if os.path.exists(PATH_TO_HULL_COLORED) is False:
        os.mkdir(PATH_TO_HULL_COLORED)

    # for idx, file_ in enumerate(os.listdir(PATH_PLYFILE)):
    for file_ in tqdm(os.listdir(PATH_CADFILE)):
        if len(file_.split('.')) == 2:
            pcd_hull = load_point_cloud(file_)
            pcd_cad = convert_mesh_to_pcd(file_,1000)
            #o3d.visualization.draw_geometries([pcd_cad])
            pcd_hull = color_pcd_hull(pcd_hull, pcd_cad, file_)
            o3d.io.write_point_cloud(PATH_TO_HULL_COLORED +"/"+ file_, pcd_hull)
        

if __name__ == "__main__":
    main()



