from pathlib import Path

import numpy as np
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from aici.tf_utils import load_tf_graph_from_bag
from aici.project_lidar_office_tf import pointcloud2_to_xyz_array


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """
    Write an ASCII PLY file with XYZ + RGB.

    xyz: (N, 3) float32
    rgb: (N, 3) uint8
    """
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def quat_to_rot(qx, qy, qz, qw):
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    """
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    x, y, z, w = qx, qy, qz, qw

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1.0 - s * (yy + zz),     s * (xy - wz),         s * (xz + wy)],
        [    s * (xy + wz),   1.0 - s * (xx + zz),       s * (yz - wx)],
        [    s * (xz - wy),       s * (yz + wx),     1.0 - s * (xx + yy)]
    ], dtype=np.float32)
    return R


def odom_msg_to_T_world_base(odom_msg) -> np.ndarray:
    """
    Convert nav_msgs/msg/Odometry to 4x4 transform from world (odom) to base_link.
    """
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation

    R = quat_to_rot(q.x, q.y, q.z, q.w)
    t = np.array([p.x, p.y, p.z], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    ROOT = Path(__file__).resolve().parents[2]
    bag_dir = ROOT / "data" / "bathroom" / "rosbag"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load camera intrinsics
    K_path = ROOT / "data" / "bathroom" / "bathroom_cam_K.npy"
    if not K_path.exists():
        raise FileNotFoundError(
            f"{K_path} not found. Please run `python3 -m aici.extract_bathroom_samples` first."
        )
    K = np.load(K_path)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print("Loaded bathroom K:\n", K)

    # 2) Load TF graph and get static extrinsics
    tf_graph = load_tf_graph_from_bag(bag_dir)
    LIDAR_FRAME = "livox_frame"
    CAM_FRAME = "zed_left_camera_optical_frame"
    BASE_FRAME = "base_link"

    # camera <- lidar
    try:
        T_cam_lidar = tf_graph.get_transform(CAM_FRAME, LIDAR_FRAME)
        print("[TF] Got T_cam_lidar (CAM_FRAME <- LIDAR_FRAME)")
    except Exception as e:
        print("[TF] Could not get T_cam_lidar directly:", e)
        print("[TF] Falling back to inverse of (LIDAR_FRAME <- CAM_FRAME)")
        T_lidar_cam = tf_graph.get_transform(LIDAR_FRAME, CAM_FRAME)
        T_cam_lidar = np.linalg.inv(T_lidar_cam)
    print("T_cam_lidar =\n", T_cam_lidar)

    # base <- lidar
    try:
        T_base_lidar = tf_graph.get_transform(BASE_FRAME, LIDAR_FRAME)
        print("[TF] Got T_base_lidar (BASE_FRAME <- LIDAR_FRAME)")
    except Exception as e:
        print("[TF] Could not get T_base_lidar directly:", e)
        print("[TF] Falling back to inverse of (LIDAR_FRAME <- BASE_FRAME)")
        T_lidar_base = tf_graph.get_transform(LIDAR_FRAME, BASE_FRAME)
        T_base_lidar = np.linalg.inv(T_lidar_base)
    print("T_base_lidar =\n", T_base_lidar)

    # 3) Open rosbag (streaming)
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}
    print("Topics in bathroom bag:")
    for name, ttype in topic_types.items():
        print(f"  {name} -> {ttype}")

    lidar_topic = "/livox/lidar"
    rgb_topic = "/zed/zed_node/rgb/image_rect_color/compressed"
    odom_topic = "/odom"

    if lidar_topic not in topic_types:
        raise RuntimeError(f"{lidar_topic} not found in bag")
    if rgb_topic not in topic_types:
        raise RuntimeError(f"{rgb_topic} not found in bag")
    if odom_topic not in topic_types:
        raise RuntimeError(f"{odom_topic} not found in bag")

    lidar_msg_type = get_message(topic_types[lidar_topic])
    rgb_msg_type = get_message(topic_types[rgb_topic])
    odom_msg_type = get_message(topic_types[odom_topic])

    # 4) Streaming state
    last_rgb_img = None
    last_T_world_base = None
    lidar_idx = 0

    STRIDE = 10          # use every 10th lidar frame
    MAX_POINTS = 300000  # global cap on points

    all_xyz_world = []
    all_rgb = []

    print("Streaming bathroom bag and building world-frame colorized cloud...")
    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == rgb_topic:
            msg = deserialize_message(data, rgb_msg_type)
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                last_rgb_img = img
            continue

        if topic == odom_topic:
            msg = deserialize_message(data, odom_msg_type)
            last_T_world_base = odom_msg_to_T_world_base(msg)
            continue

        if topic == lidar_topic:
            lidar_idx += 1
            if lidar_idx % STRIDE != 0:
                continue
            if last_rgb_img is None or last_T_world_base is None:
                # need both an image and a pose to use this cloud
                continue

            msg = deserialize_message(data, lidar_msg_type)
            pts_lidar = pointcloud2_to_xyz_array(msg)  # (N, 3)
            if pts_lidar.size == 0:
                continue

            img = last_rgb_img
            H, W, _ = img.shape

            # --- 4.a) Colorization: lidar -> cam, project to image ---
            N = pts_lidar.shape[0]
            pts_lidar_h = np.hstack(
                [pts_lidar.astype(np.float32), np.ones((N, 1), dtype=np.float32)]
            )
            pts_cam_h = (T_cam_lidar @ pts_lidar_h.T).T
            pts_cam = pts_cam_h[:, :3]

            Z = pts_cam[:, 2]
            valid_z = Z > 0.1
            if not np.any(valid_z):
                continue

            X = pts_cam[valid_z, 0]
            Y = pts_cam[valid_z, 1]
            Z_valid = pts_cam[valid_z, 2]

            u = (fx * X / Z_valid + cx).astype(np.int32)
            v = (fy * Y / Z_valid + cy).astype(np.int32)

            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not np.any(inside):
                continue

            u = u[inside]
            v = v[inside]

            idx_all = np.where(valid_z)[0][inside]

            # --- 4.b) Geometry: lidar -> world using odom + base <- lidar ---
            T_world_lidar = last_T_world_base @ T_base_lidar  # world <- lidar
            pts_lidar_sel = pts_lidar[idx_all]  # (M, 3)
            M = pts_lidar_sel.shape[0]
            pts_lidar_sel_h = np.hstack(
                [pts_lidar_sel.astype(np.float32), np.ones((M, 1), dtype=np.float32)]
            )
            pts_world_h = (T_world_lidar @ pts_lidar_sel_h.T).T
            pts_world = pts_world_h[:, :3]

            # --- 4.c) Colors from image (RGB) ---
            colors_bgr = img[v, u, :]
            colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

            all_xyz_world.append(pts_world)
            all_rgb.append(colors_rgb)

            print(f"[Bathroom LIDAR frame {lidar_idx}] kept {pts_world.shape[0]} world-frame colored points.")

            current_total = sum(chunk.shape[0] for chunk in all_xyz_world)
            if current_total >= MAX_POINTS:
                print(f"Reached MAX_POINTS={MAX_POINTS}, stopping early.")
                break

    if not all_xyz_world:
        raise RuntimeError("No colored world points collected for bathroom. Check topics, transforms, and STRIDE.")

    all_xyz_world = np.vstack(all_xyz_world).astype(np.float32)
    all_rgb = np.vstack(all_rgb).astype(np.uint8)

    print(f"Final bathroom world-frame concatenated cloud: {all_xyz_world.shape[0]} points.")

    out_ply = results_dir / "bathroom_colorized_cloud_world.ply"
    write_ply_xyzrgb(out_ply, all_xyz_world, all_rgb)
    print(f"Saved bathroom world-frame colorized concatenated cloud to: {out_ply}")


if __name__ == "__main__":
    main()
