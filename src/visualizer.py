from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


my_data = genfromtxt('data/voxelFile.csv', delimiter=',')



x = np.array(my_data[1:, 0])
y = np.array(my_data[1:, 1]) 
z = np.array(my_data[1:, 2]) 



voxels = np.transpose(np.vstack((x,y,z)))

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')
colors = np.array(my_data[1:, 3])
print(voxels)
ma = np.random.choice([0,1], size=(10,10,10), p=[0.99, 0.01])
#print(ma)
ax.scatter3D(x,y,z,c=colors, cmap = 'Greens')

plt.show()