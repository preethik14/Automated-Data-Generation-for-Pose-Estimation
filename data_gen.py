import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.spatial import KDTree

# Setup
output_dir = 'synthetic_dataset'
os.makedirs(output_dir, exist_ok=True)

# Load CAD model and apply scale
model = trimesh.load("/home/rbccps/diadem2.glb")
mesh = trimesh.util.concatenate(model.dump())
mesh.apply_scale(0.1)
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle=np.radians(135),     # angle in radians
    direction=[0, 1, 0],      # Y axis
    point=[0, 0, 0]           # rotation center (origin)
)

mesh.apply_transform(rotation_matrix)
# Load textured quad
def load_quad_with_texture(path, size=96.0):
    image = Image.open(path).convert("RGB").resize((512, 512))
    half = size / 2
    vertices = np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0]
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    material = trimesh.visual.texture.SimpleMaterial(image=image)
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=image, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)

quad = load_quad_with_texture("/media/rbccps/HDD/cad/000000000143.jpg")
rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])  # 90 deg around Y
quad.apply_transform(rotation)

# Move quad behind robot, aligned with robot on Y/Z
quad.apply_translation([-20, 0, 0])  # behind robot at -5 on X
# Create scene
scene = trimesh.Scene()
scene.add_geometry(quad)

# Setup camera looking in -X direction
camera = trimesh.scene.Camera(resolution=(640, 480), fov=(110, 110))
scene.camera = camera
rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
scene.camera_transform = rotation
camera_transform = trimesh.transformations.rotation_matrix(
    angle=np.pi / 2,  # +90 deg
    direction=[0, 1, 0],  # rotate around Y
    point=[0, 0, 0]
)
scene.camera_transform = camera_transform
# Move robot mesh into view
mesh.apply_translation([-15, 0, 0])
scene.add_geometry(mesh)

# --- Harris 3D Keypoints
def harris_3d_keypoints(mesh: trimesh.Trimesh, k_neighbors=10, harris_k=0.04, threshold_percentile=97):
    mesh = mesh.subdivide()
    vertices = np.array(mesh.vertices)
    kdtree = KDTree(vertices)

    def get_local_points(vertex_index, k=k_neighbors):
        _, idx = kdtree.query(vertices[vertex_index], k=k)
        return vertices[idx]

    def compute_structure_tensor(points):
        mean = np.mean(points, axis=0)
        cov_matrix = np.cov((points - mean).T)
        return cov_matrix

    R_values = []
    for i in range(len(vertices)):
        local_pts = get_local_points(i)
        eigenvalues, _ = np.linalg.eigh(compute_structure_tensor(local_pts))
        R = eigenvalues[0] * eigenvalues[1] - harris_k * (eigenvalues[0] + eigenvalues[1]) ** 2
        R_values.append(R)

    R_values = np.array(R_values)
    threshold = np.percentile(R_values, threshold_percentile)
    keypoints = vertices[R_values > threshold]
    return keypoints, R_values

keypoints, R_vals = harris_3d_keypoints(mesh)
print(f"Detected {len(keypoints)} Harris 3D keypoints")

# --- Project 3D keypoints to 2D
def project_points(points, camera, cam_transform, width, height):
    # Intrinsics
    fov_x = np.radians(camera.fov[0])
    fx = fy = width / (2 * np.tan(fov_x / 2))
    cx, cy = width / 2, height / 2

    # Transform points into camera space
    T = np.linalg.inv(cam_transform)
    points_cam = trimesh.transformations.transform_points(points, T)

    projected = []
    for p in points_cam:
        if p[0] >= 0:  # Points behind the camera in -X
            continue
        x = fx * p[1] / -p[0] + cx
        y = fy * p[2] / -p[0] + cy
        projected.append([x, y])

    return np.array(projected)

projected_kps = project_points(keypoints, camera, scene.camera_transform, 640, 480)

# --- Bounding box from projected keypoints
if len(projected_kps) > 0:
    x_min, y_min = np.min(projected_kps, axis=0)
    x_max, y_max = np.max(projected_kps, axis=0)
    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
else:
    bbox = [0, 0, 0, 0]  # Fallback in case of no visible keypoints
scene.show()
# --- Save image
image_bytes = scene.save_image(resolution=(640, 480), visible=True)
image = Image.open(trimesh.util.wrap_as_stream(image_bytes))
image_path = os.path.join(output_dir, "image_0001.png")
image.save(image_path)

# --- Save label file
label_path = os.path.join(output_dir, "image_0001.txt")
with open(label_path, 'w') as f:
    f.write(f"# BBox: x_min y_min x_max y_max\n")
    f.write("bbox: " + " ".join(map(str, bbox)) + "\n")
    f.write("# Keypoints: x y\n")
    for kp in projected_kps:
        f.write("keypoint: {:.2f} {:.2f}\n".format(kp[0], kp[1]))

# --- Visualize
plt.imshow(image)
if len(projected_kps) > 0:
    plt.scatter(projected_kps[:, 0], projected_kps[:, 1], c='r', s=5)
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                      edgecolor='lime', linewidth=1, fill=False))
plt.title("Projected Keypoints + Bounding Box")
plt.axis('off')
plt.show()
