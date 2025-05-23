import trimesh
import numpy as np
from PIL import Image
import os
from scipy.spatial import KDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector

# ---------- Directory Setup ----------
base_dir = 'synthetic_dataset'
splits = ['train', 'val']
for split in splits:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# ---------- Constants ----------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
N_FIXED = 500
ROTATION_ANGLES = np.arange(0, 361, 20)
MAX_IMAGES = 40000
IMAGES_PER_POSITION = 5

# ---------- Load Model ----------
model = trimesh.load("/home/rbccps/diadem2.glb")
mesh_base = trimesh.util.concatenate(model.dump())
mesh_base.apply_scale(0.1)
intersector = RayMeshIntersector(mesh_base)

# ---------- Utilities ----------
def get_texture_paths(folder_path, extensions=(".jpg", ".png", ".jpeg")):
    return [os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(extensions)]

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

def filter_occluded_keypoints(keypoints, directions, intersector, threshold=1):
    visible_mask = np.zeros(len(keypoints), dtype=int)
    for d in directions:
        ray_origins = keypoints + d * 1e-2
        ray_dirs = -np.tile(d, (len(keypoints), 1))
        hits = intersector.intersects_any(ray_origins, ray_dirs)
        visible_mask += ~hits
    return keypoints[visible_mask >= threshold]

def harris_3d_keypoints(mesh, k_neighbors=20, harris_k=0.04, target_count=50):
    mesh = mesh.subdivide()
    vertices = np.array(mesh.vertices)
    kdtree = KDTree(vertices)
    R_values = []

    for i in range(len(vertices)):
        _, idx = kdtree.query(vertices[i], k=k_neighbors)
        local_pts = vertices[idx]
        cov = np.cov((local_pts - np.mean(local_pts, axis=0)).T)
        eigs = np.linalg.eigvalsh(cov)
        R = eigs[0] * eigs[1] - harris_k * (eigs[0] + eigs[1]) ** 2
        R_values.append(R)

    R_values = np.array(R_values)

    # Dynamically find the threshold to get ~target_count keypoints
    percentiles = np.linspace(99, 80, 100)
    for p in percentiles:
        threshold = np.percentile(R_values, p)
        keypoints = vertices[R_values > threshold]
        if len(keypoints) >= target_count:
            break

    print(f"Selected {len(keypoints)} keypoints at threshold percentile: {p}")
    directions = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    return filter_occluded_keypoints(keypoints, directions, intersector)

def project_points(points, camera, cam_transform, width, height):
    fov_x = np.radians(camera.fov[0])
    fx = fy = width / (2 * np.tan(fov_x / 2))
    cx, cy = width / 2, height / 2
    T = np.linalg.inv(cam_transform)
    points_cam = trimesh.transformations.transform_points(points, T)
    projected = []
    for p in points_cam:
        if p[2] >= 0: continue
        x = fx * p[0] / -p[2] + cx
        y = fy * p[1] / -p[2] + cy
        projected.append([x, height - y])
    return np.array(projected)
    

def compute_visibility(points_3d, intersector, cam_transform, epsilon=1.0):
    """
    Returns visibility array: 2 if visible, 0 if occluded.
    Shoots rays from each 3D keypoint toward the camera origin.
    """
    cam_pos = trimesh.transformations.transform_points([[0, 0, 0]], np.linalg.inv(cam_transform))[0]
    directions = cam_pos - points_3d
    distances = np.linalg.norm(directions, axis=1)
    directions /= distances[:, np.newaxis]

    origins = points_3d + directions * 1e-3
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    visible_mask = np.ones(len(points_3d), dtype=int) * 2

    for i in range(len(points_3d)):
        matches = np.where(index_ray == i)[0]
        if len(matches) == 0:
            continue
        hit_point = locations[matches[0]]
        hit_distance = np.linalg.norm(hit_point - points_3d[i])
        if hit_distance < distances[i] - epsilon:
            visible_mask[i] = 0
    return visible_mask



def add_salt_and_pepper_noise(image, amount=0.02):
    arr = np.array(image)
    num_salt = int(np.ceil(amount * arr.size * 0.5))
    num_pepper = int(np.ceil(amount * arr.size * 0.5))
    coords = [np.random.randint(0, i, num_salt) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, num_pepper) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)

# ---------- Setup ----------
camera = trimesh.scene.Camera(resolution=(IMAGE_WIDTH, IMAGE_HEIGHT), fov=(110, 110))
camera_transform = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
cam_origin = trimesh.transformations.transform_points([[0, 0, 0]], np.linalg.inv(camera_transform))[0]

# ---------- Keypoints ----------
keypoints = harris_3d_keypoints(mesh_base)
print(f"Total 3D keypoints: {len(keypoints)}")
fixed_keypoints_3d = keypoints[np.linspace(0, len(keypoints)-1, N_FIXED, dtype=int)]
np.save("keypoints_3d.npy", fixed_keypoints_3d)
fixed_keypoints_3d = np.load("keypoints_3d.npy")

# ---------- Dataset Loop ----------
texture_paths = get_texture_paths("/home/rbccps/voltas/Assets/Backgrounds")
print(f"Found {len(texture_paths)} textures.")
y_values = np.arange(-6, 7, 1)
z_values = np.arange(-6, 7, 1)
positions = [(y, z) for y in y_values for z in z_values]
position_index = 0
position_image_count = 0
image_counter = 0

while image_counter < MAX_IMAGES:
    for texture_path in texture_paths:
        if image_counter >= MAX_IMAGES:
            break
        for angle in ROTATION_ANGLES:
            if image_counter >= MAX_IMAGES:
                break
            if position_image_count == 0:
                position_index = (position_index + 1) % len(positions)
                y, z = positions[position_index]
                robot_position = np.array([-15, y, z])
                print(f"\n--- New Robot Position: {robot_position} (#{position_index + 1}) ---")

            scene = trimesh.Scene()
            quad = load_quad_with_texture(texture_path)
            quad.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
            quad.apply_translation([-20, 0, 0])
            scene.add_geometry(quad)

            scene.camera = camera
            scene.camera_transform = camera_transform

            mesh = mesh_base.copy()
            center_transform = trimesh.transformations.translation_matrix(-mesh.centroid)
            rotation_transform = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
            translation_transform = trimesh.transformations.translation_matrix(robot_position)
            robot_transform = translation_transform @ rotation_transform @ center_transform

            mesh = mesh_base.copy()
            mesh.apply_transform(robot_transform)
            intersector = RayMeshIntersector(mesh)

            scene.add_geometry(mesh)
            

            keypoints_transformed = trimesh.transformations.transform_points(fixed_keypoints_3d, robot_transform)
            projected = project_points(keypoints_transformed, camera, camera_transform, IMAGE_WIDTH, IMAGE_HEIGHT)
            in_frame = (projected[:,0] >= 0) & (projected[:,0] <= IMAGE_WIDTH) & \
                       (projected[:,1] >= 0) & (projected[:,1] <= IMAGE_HEIGHT)
            keypoints_transformed = keypoints_transformed[in_frame]
            projected = projected[in_frame]

            vertices = np.asarray(mesh.vertices)
            vertices_transformed = trimesh.transformations.transform_points(vertices, robot_transform)

            # Step 2: Ray-based visibility check (optional but recommended)
            visible_mask = compute_visibility(keypoints_transformed, intersector, camera_transform)  # same as for keypoints
            visible_vertices = vertices_transformed[visible_mask]

            # Step 3: Project visible vertices to image plane
            projected_verts = project_points(vertices, camera, camera_transform, IMAGE_WIDTH, IMAGE_HEIGHT)

            if projected_verts.shape[0] > 0 and projected_verts.shape[1] == 2:
                in_frame_verts = np.all((projected_verts >= 0) & (projected_verts < [IMAGE_WIDTH, IMAGE_HEIGHT]), axis=1)
                projected_verts = projected_verts[in_frame_verts]

                if len(projected_verts) > 0:
                    x_min, y_min = np.min(projected_verts, axis=0)
                    x_max, y_max = np.max(projected_verts, axis=0)
                    cx, cy = ((x_min + x_max) / 2) / IMAGE_WIDTH, ((y_min + y_max) / 2) / IMAGE_HEIGHT
                    w, h = (x_max - x_min) / IMAGE_WIDTH, (y_max - y_min) / IMAGE_HEIGHT
                else:
                    cx = cy = w = h = 0
            else:
                cx = cy = w = h = 0

            if image_counter < 20000:
                split, noise = 'train', False
            elif image_counter < 25000:
                split, noise = 'train', True
            elif image_counter < 35000:
                split, noise = 'val', False
            else:
                split, noise = 'val', True

            # Visualize keypoints
            kps_spheres = trimesh.points.PointCloud(keypoints_transformed, colors=[255, 0, 0])
            # scene.add_geometry(kps_spheres)
            img_bytes = scene.save_image(resolution=(IMAGE_WIDTH, IMAGE_HEIGHT), visible=True)
            image = Image.open(trimesh.util.wrap_as_stream(img_bytes))
            if noise:
                image = add_salt_and_pepper_noise(image)

            img_name = f"image_{image_counter:05d}.png"
            image.save(os.path.join(base_dir, split, 'images', img_name))

            label_path = os.path.join(base_dir, split, 'labels', f"image_{image_counter:05d}.txt")
            with open(label_path, 'w') as f:
                f.write(f"0 {cx:.7f} {cy:.7f} {w:.7f} {h:.7f} ")
                visibilities = compute_visibility(keypoints_transformed, intersector, camera_transform)
                for (kp_2d, vis) in zip(projected, visibilities):
                    x_norm = kp_2d[0] / IMAGE_WIDTH
                    y_norm = kp_2d[1] / IMAGE_HEIGHT
                    f.write(f"{x_norm:.7f} {y_norm:.7f} {vis} ")

            print(f"[{image_counter+1}/{MAX_IMAGES}] {split.upper()} -> {img_name}")
            image_counter += 1
            position_image_count += 1
            if position_image_count >= IMAGES_PER_POSITION:
                position_image_count = 0
