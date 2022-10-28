
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

#proporition_sampling = 0.05 # proportion of sampling points
#pcd = pcd.uniform_down_sample(every_k_points=int(1/proporition_sampling))
#print("Number of points of the resampled cloud:", len(pcd.points))"""

""" Algorithms: RANSAC (RANdom SAmple Consensus) implementation for planar shape detection in point clouds 
    + DBSCAN (Euclidean Clustering (DBSCAN) for clustering points """

segment_models = {} # empty dictionary that will hold the results of the iterations
segments = {} # (the plane parameters in segment_models, and the planar regions from the point cloud in segments)
max_plane_idx = 20 # number of iterations (here we want to iterate 20 times to find 20 planes)
rest = pcd
d_threshold = 0.01
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000) # 3 parameters: distance threshold from the plane to consider a point inlier or outlier, the number of sampled points drawn (3 here, as we want a plane) to estimate each plane candidate (ransac_n) and the number of iterations (num_iterations). 
    segments[i] = rest.select_by_index(inliers) # store the inliers in segments
    labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=10)) # run DBSCAN just after the assignment of the inliers
    candidates = [len(np.where(labels==j)[0]) for j in np.unique(labels)] # how many points each cluster that we found holds
    best_candidate = int(np.unique(labels)[np.where(candidates==np.max(candidates))[0]]) # find the “best candidate”, which is normally the cluster that holds the more points
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
