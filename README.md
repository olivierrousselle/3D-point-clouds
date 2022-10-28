# Point-clouds
This code implements tools/algorithms to perform the treatement of point clouds. In particular: Computation of octrees from the point cloud (file point_cloud_octree); 
those octrees can be used for a lot of tasks in 3D rendering and spatial partioning; Segmentation of the point cloud (file point_cloud_segmentation) with RANSAC and DBSCAN methods. 
The visualization of the point clouds and other shapes is done by using Open3D. Moreover, the loops "for" are accelerated by using numba and parallelization on CPUs. 
