from pathlib import Path

import numpy as np
import cv2

from sensor_msgs.msg import PointCloud2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from aici.tf_utils import load_tf_graph_from_bag


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 to (N,3) xyz array."""
    from sensor_msgs_py import point_cloud2
    points = []
    for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def main():
    ROOT = Path(__file__).resolve().parents[2]
    bag_dir = ROOT / "data" / "office" / "rosbag"
    img_path = ROOT / "data" / "office" / "office_rgb_sample.png"
    K_path = ROOT / "data" / "office" / "office_cam_K.npy"

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")
    K = np.load(K_path)

    print("Image shape:", img.shape)
    print("Camera intrinsics K:\n", K)

    # 1) Build TF graph
    tf_graph = load_tf_graph_from_bag(bag_dir)
    LIDAR_FRAME = "livox_frame"
    CAM_FRAME = "zed_left_camera_optical_frame"

    # --- Get LiDAR -> camera transform (T_cam_lidar) correctly ---
    # We want: p_cam = T_cam_lidar @ p_lidar_h
    try:
        # Try to get transform directly as camera <- lidar
        T_cam_lidar = tf_graph.get_transform(CAM_FRAME, LIDAR_FRAME)
        print("[TF] Got T_cam_lidar as get_transform(CAM_FRAME, LIDAR_FRAME)")
    except Exception as e:
        print("[TF] Could not get (CAM_FRAME, LIDAR_FRAME) directly:", e)
        print("[TF] Falling back to get_transform(LIDAR_FRAME, CAM_FRAME) and inverting.")
        T_lidar_cam = tf_graph.get_transform(LIDAR_FRAME, CAM_FRAME)
        T_cam_lidar = np.linalg.inv(T_lidar_cam)

    print("T_cam_lidar =\n", T_cam_lidar)

    # 2) Open rosbag and read one LiDAR message
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}

    lidar_topic = "/livox/lidar"
    if lidar_topic not in topic_types:
        raise RuntimeError(f"{lidar_topic} not found in bag")

    lidar_msg_type = get_message(topic_types[lidar_topic])

    cloud_xyz_lidar = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != lidar_topic:
            continue
        msg = deserialize_message(data, lidar_msg_type)  # already a PointCloud2
        cloud_xyz_lidar = pointcloud2_to_xyz_array(msg)
        break

    if cloud_xyz_lidar is None:
        raise RuntimeError("No LiDAR cloud read from bag")

    print("LiDAR points (lidar frame):", cloud_xyz_lidar.shape)

    # 3) Transform LiDAR -> camera frame
    N = cloud_xyz_lidar.shape[0]
    pts_lidar_h = np.hstack(
        [cloud_xyz_lidar, np.ones((N, 1), dtype=np.float32)]
    )  # (N, 4)
    pts_cam_h = (T_cam_lidar @ pts_lidar_h.T).T  # (N, 4)
    pts_cam = pts_cam_h[:, :3]

    # Debug: show a few transformed points
    print("Example LiDAR -> camera points:")
    for i in range(min(5, N)):
        print("  lidar:", cloud_xyz_lidar[i], " -> cam:", pts_cam[i])

    # 4) Project to image with K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    Z = pts_cam[:, 2]
    valid = Z > 0.1  # keep only points in front of camera

    X = pts_cam[valid, 0]
    Y = pts_cam[valid, 1]
    Z = pts_cam[valid, 2]

    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)

    H, W, _ = img.shape
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inside]
    v = v[inside]

    print(f"Projected {u.size} LiDAR points into the image.")

    img_overlay = img.copy()
    for uu, vv in zip(u, v):
        cv2.circle(img_overlay, (uu, vv), 1, (0, 0, 255), -1)

    out_path = ROOT / "results" / "office_rgb_lidar_overlay_tf.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_overlay)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
