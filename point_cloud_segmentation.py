# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:56:47 2022

@author: Olivier
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numba as nb
import time

start_time = time.time()

""" Opening and vizualization of the point cloud """

point_cloud = np.loadtxt("the_researcher_desk.xyz",skiprows=1)
mean_Z = np.mean(point_cloud,axis=0)[2]
spatial_query = point_cloud[abs( point_cloud[:,2]-mean_Z)<1]
xyz, rgb = spatial_query[:,:3], spatial_query[:,3:]
print("Cloud file read")
pcd = o3d.geometry.PointCloud()
pcd.points, pcd.colors = o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3dVector(rgb/255)
print("Number of input points:", len(pcd.points))
#o3d.visualization.draw_geometries([pc])

#pc = o3d.io.read_point_cloud("TLS_kitchen.ply")
"""proporition_sampling = 0.1 # proportion of sampling points
pcd = pcd.uniform_down_sample(every_k_points=int(1/proporition_sampling))
print("Number of points of the resampled cloud:", len(pcd.points))"""
#o3d.visualization.draw_geometries([pcd])

pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)
min_x, min_y, min_z = pcd.get_min_bound()
max_x, max_y, max_z = pcd.get_max_bound()
x_range, y_range, z_range = max_x - min_x, max_y - min_y, max_z - min_z

""" RANSAC (RANdom SAmple Consensus) """

# RANSAC implementation for planar shape detection in point clouds
# 3 parameters: distance threshold from the plane to consider a point inlier or outlier, the number of sampled points drawn (3 here, as we want a plane) to estimate each plane candidate (ransac_n) and the number of iterations (num_iterations). 
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

""" Euclidean Clustering (DBSCAN) """

# cluster_dbscan acts on the point cloud entity directly and returns a list of labels following the initial indexing of the point cloud.
# radius of 5 cm for “growing” clusters, and considering one only if after this step we have at least 10 points
# the labels vary between -1 and n, where -1 indicate it is a “noise” point and values 0 to n are then the cluster labels given to the corresponding point.
labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=10))
# color the results
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0 # make sure to make noisy points to black
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#o3d.visualization.draw_geometries([pcd])

""" RANSAC loop for multiple planar shapes detection """

segment_models = {} # empty dictionary that will hold the results of the iterations
segments = {} # (the plane parameters in segment_models, and the planar regions from the point cloud in segments)
max_plane_idx = 20 # number of iterations (here we want to iterate 20 times to find 20 planes)
rest=pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    segments[i] = rest.select_by_index(inliers) # store the inliers in segments
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True) # pursue with only the remaining points stored in rest, that becomes the subject of interest for the loop n+1
    print("pass",i,"/",max_plane_idx,"done.")
#o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])


""" Wise refinement of the multi-RANSAC loop with DBSCAN """

segment_models = {} # empty dictionary that will hold the results of the iterations
segments = {} # (the plane parameters in segment_models, and the planar regions from the point cloud in segments)
max_plane_idx = 20 # number of iterations (here we want to iterate 20 times to find 20 planes)
rest = pcd
d_threshold = 0.01
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
    segments[i] = rest.select_by_index(inliers) # store the inliers in segments
    labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=10)) # run DBSCAN just after the assignment of the inliers
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)] # how many points each cluster that we found holds
    best_candidate=int(np.unique(labels)[np.where(candidates==np.max(candidates))[0]]) # find the “best candidate”, which is normally the cluster that holds the more points
    print("the best candidate is: ", best_candidate)
    rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i] = segments[i].select_by_index(list(np.where(labels==best_candidate)[0]))
    segments[i].paint_uniform_color(list(colors[:3]))
    #o3d.visualization.draw_geometries([segments[i]])
    print("pass",i+1,"/",max_plane_idx,"done.")
    
labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {len(segments) + max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

print("Time execution full process: %s seconds ---" % round(time.time() - start_time)) 

# o3d.visualization.draw_geometries([segments.values()])
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
# o3d.visualization.draw_geometries([rest])
