# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:28:04 2022

@author: Olivier
"""

import numpy as np
import open3d as o3d
import time
import numba as nb

# nb.jit(parallel=True, fastmath=True)
print("Number of threads: %s" % nb.get_num_threads())


""" Opening and vizualization of the point cloud """

point_cloud = np.loadtxt("the_researcher_desk.xyz",skiprows=1)
mean_Z = np.mean(point_cloud,axis=0)[2]
spatial_query = point_cloud[abs( point_cloud[:,2]-mean_Z)<1]
xyz, rgb = spatial_query[:,:3], spatial_query[:,3:]
print("Cloud file read")
pc = o3d.geometry.PointCloud()
pc.points, pc.colors = o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3dVector(rgb/255)
print("Number of input points:", len(pc.points))
#o3d.visualization.draw_geometries([pc])


start_time = time.time()

""" Sampling """

proporition_sampling = 0.05 # proportion of sampling points
pc_rs = pc.uniform_down_sample(every_k_points=int(1/proporition_sampling))
print("Number of points of the resampled cloud:", len(pc_rs.points))
#o3d.visualization.draw_geometries([pc_rs])
xyz = np.asarray(pc_rs.points)
rgb = np.asarray(pc_rs.colors)

xyzmin = np.min(xyz,axis=0)
xyzmax = np.max(xyz,axis=0)
min_x, min_y, min_z = xyzmin[0], xyzmin[1], xyzmin[2] 
max_x, max_y, max_z = xyzmax[0], xyzmax[1], xyzmax[2] 
x_range, y_range, z_range = max_x - min_x, max_y - min_y, max_z - min_z
num_levels_octree = 5
grid_size = x_range/(2**num_levels_octree)
x_idx, y_idx, z_idx = int(max(1, np.ceil(x_range/grid_size))), int(max(1, np.ceil(y_range/grid_size))), int(max(1, np.ceil(z_range/grid_size)))
num_voxels = x_idx * y_idx * z_idx
print("Number of voxels:", num_voxels)

nodes_ini = np.array([[[min_x+grid_size*i, min_y+grid_size*j, min_z+grid_size*k], [min_x+grid_size*(i+1), min_y+grid_size*(j+1), min_z+grid_size*(k+1)]] for i in range(x_idx) for j in range(y_idx) for k in range(z_idx)])
xyz = np.asarray(pc_rs.points)

@nb.jit(parallel=True)
def compute_nodes_points(xyz, nodes_ini):
    nodes = []
    nodes_points = []
    for j in nb.prange(len(nodes_ini)):
        condition = np.array((xyz[:,0] >= nodes_ini[j,0,0]) & (xyz[:,1] >= nodes_ini[j,0,1]) & (xyz[:,2] >= nodes_ini[j,0,2])
                   & (xyz[:,0] < nodes_ini[j,1,0]) & (xyz[:,1] < nodes_ini[j,1,1]) & (xyz[:,2] < nodes_ini[j,1,2]))
        xyz_selected = xyz[condition]
        if len(xyz_selected) > 0:
            xyz = xyz[~condition]
            nodes.append(nodes_ini[j])
            nodes_points.append(xyz_selected)
    return nodes, nodes_points

nodes, nodes_points = compute_nodes_points(xyz, nodes_ini)

mat_line_set = []
for j in range(len(nodes)):
    x_node, y_node, z_node = [nodes[j][0,0], nodes[j][1,0]], [nodes[j][0,1], nodes[j][1,1]], [nodes[j][0,2], nodes[j][1,2]]
    points = np.array(list(product(x_node, y_node, z_node)))
    lines = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7]]) 
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    mat_line_set.append(line_set)
    
print("Time execution full process: %s seconds ---" % round(time.time() - start_time)) 

o3d.visualization.draw_geometries([pc_rs]+[mat_line_set[i] for i in range(len(mat_line_set))])
