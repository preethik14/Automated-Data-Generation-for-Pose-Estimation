import trimesh
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

# Load and preprocess the mesh
mesh1 = trimesh.load("/home/rbccps/diadem2.glb")
if isinstance(mesh1, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh1.dump())
else:
    mesh = mesh1

print("Is mesh watertight?", mesh.is_watertight)

# mesh = mesh.subdivide()  # Subdividing before extracting vertices
# vertices = np.array(mesh.vertices)
# kdtree = KDTree(vertices)
# print(mesh.visual.material)

# # Compute vertex normals
# normals = np.array(mesh.vertex_normals)

# # Function to get k-nearest neighbors of a vertex
# def get_local_points(vertex_index, k=10):
#     _, idx = kdtree.query(vertices[vertex_index], k=k)
#     return vertices[idx]

# # Compute structure tensor (covariance matrix)
# def compute_structure_tensor(points):
#     mean = np.mean(points, axis=0)
#     cov_matrix = np.cov((points - mean).T)
#     return cov_matrix

# # Compute R values for all vertices
# k = 0.04
# R_values = []

# for i in range(len(vertices)):
#     local_pts = get_local_points(i, k=20)
#     eigenvalues, _ = np.linalg.eigh(compute_structure_tensor(local_pts))
#     R = eigenvalues[0] * eigenvalues[1] - k * (eigenvalues[0] + eigenvalues[1]) ** 2
#     R_values.append(R)

# R_values = np.array(R_values)
# threshold = np.percentile(R_values, 97)
# keypoints = vertices[R_values > threshold]

# print(keypoints)

# # Visualizing with Open3D
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices)
# pcd.paint_uniform_color([1, 1, 1])


# mesh_o3d = o3d.geometry.TriangleMesh()
# mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
# mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
# if hasattr(mesh.visual, 'vertex_colors'):
#     vertex_colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0  # Normalize to [0,1]
#     mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
# mesh_o3d.vertex_normals  # Ensures smooth shading
# print(mesh.visual.material)  # If this exists, the GLB has vertex colors
# # mesh.show(viewer='gl')



# keypoint_pcd = o3d.geometry.PointCloud()
# keypoint_pcd.points = o3d.utility.Vector3dVector(keypoints)
# keypoint_pcd.paint_uniform_color([1, 0, 0])

# o3d.visualization.draw_geometries([mesh_o3d, keypoint_pcd], mesh_show_back_face=True)

# ##freeze this code