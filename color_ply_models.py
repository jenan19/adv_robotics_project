import copy
import torch
import numpy as np
import math
import open3d as o3d
from scipy.spatial import KDTree

fileName = 'Transformer-Pulse-Electronics-PE-67050NL-PE-67100NL-PE-67200NL-PE-67300NL-wm'

pcd_hull = o3d.io.read_point_cloud("./plyFiles/"+fileName+".ply")

mesh_cad = o3d.io.read_triangle_mesh("./OGPlyFiles/"+fileName+".ply")

pcd_cad = mesh_cad.sample_points_uniformly(number_of_points=10000)


pcd_hull.scale(1000,[0,0,0])

Rx = np.matrix([[ 1, 0           , 0           ],
           [ 0, math.cos(math.pi),-math.sin(math.pi)],
           [ 0, math.sin(math.pi), math.cos(math.pi)]])


pcd_hull = pcd_hull.rotate(Rx,[0,0,0])


file1 = open('./translation/'+fileName+'/'+fileName+'_translation.txt', 'r')

translation = (file1.readlines())[0].split(" ")
translation = [float(translation[0])*1000,float(translation[1])*1000,float(translation[2])*1000]

file1.close

pcd_hull = pcd_hull.translate(translation)


#o3d.visualization.draw_geometries([pcd_hull,pcd_cad ])
o3d.visualization.draw_geometries([pcd_hull])


xyz_hull = np.asarray(pcd_hull.points)
xyz_cad = np.asarray(pcd_cad.points)

color = np.asarray(pcd_cad.colors)

kdtree=KDTree(xyz_cad)

dist,points=kdtree.query(xyz_hull,1) 


print(color)