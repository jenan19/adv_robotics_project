from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import mcubes

voxel3D = genfromtxt('build/voxelFile.csv', delimiter=',')



error_amount = 5
maxv = np.max(voxel3D[1:,3])

iso_value = maxv-np.round(((maxv)/100)*error_amount)-0.5
print('max number of votes:' + str(maxv))
print('threshold for marching cube:' + str(iso_value))
print('voxel shape: ', voxel3D.shape)

points_x = np.unique(np.array(voxel3D[1:, 0]))
points_y = np.unique(np.array(voxel3D[1:, 1]))
points_z = np.unique(np.array(voxel3D[1:, 2]))

#print('points_x: ',points_x)
voxel3D_score = np.zeros((points_x.shape[0], points_y.shape[0], points_z.shape[0]))

data_idx = 1
for x in range(voxel3D_score.shape[0]):
    for y in range(voxel3D_score.shape[1]):
        for z in range(voxel3D_score.shape[2]):
            voxel3D_score[x,y,z] = voxel3D[data_idx, 3]
            data_idx +=1




# Extract the 0-isosurface
vertices, triangles = mcubes.marching_cubes(voxel3D_score, iso_value)

# Export the result to sphere.dae
mcubes.export_mesh(vertices, triangles, "DinoR_21.dae", "DinoR_21_")





"""x = np.array(voxel3D[1:, 0])
y = np.array(voxel3D[1:, 1]) 
z = np.array(voxel3D[1:, 2]) 



voxels = np.transpose(np.vstack((x,y,z)))

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')
colors = np.array(my_data[1:, 3])
print(voxels)
ma = np.random.choice([0,1], size=(10,10,10), p=[0.99, 0.01])
#print(ma)
ax.scatter3D(x,y,z,c=colors, cmap = 'Greens')

plt.show()"""