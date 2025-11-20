from pathlib import Path

import numpy as np
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2, CompressedImage, CameraInfo
from sensor_msgs_py import point_cloud2

from .tf_utils import load_tf_graph_from_bag


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 to (N, 3) xyz array."""
    points = []
    for p in point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    ):
        points.append([p[0], p[1], p[2]])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    """Decode sensor_msgs/CompressedImage to BGR cv2 image."""
    img_np = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img


def main():
    ROOT = Path(__file__).resolve().parents[2]
    bag_dir = ROOT / "data" / "office" / "rosbag"

    out_img_path = ROOT / "data" / "office" / "office_rgb_sample.png"
    out_cloud_path = ROOT / "data" / "office" / "office_cloud_sample.npy"
    out_K_path = ROOT / "data" / "office" / "office_cam_K.npy"

    
    # 0) Choose which synchronized sample to use
  
    # Valid range (for current office bag): 0 .. 3583
    TARGET_SAMPLE_INDEX = 2034

   
    # 1) Build TF graph and get LiDAR -> camera transform
   
    tf_graph = load_tf_graph_from_bag(bag_dir)
    LIDAR_FRAME = "livox_frame"
    CAM_FRAME = "zed_left_camera_optical_frame"

    try:
        T_cam_lidar = tf_graph.get_transform(CAM_FRAME, LIDAR_FRAME)
        print("[TF] Got T_cam_lidar as get_transform(CAM_FRAME, LIDAR_FRAME)")
    except Exception as e:
        print("[TF] Could not get (CAM_FRAME, LIDAR_FRAME) directly:", e)
        print("[TF] Falling back to (LIDAR_FRAME, CAM_FRAME) and inverting.")
        T_lidar_cam = tf_graph.get_transform(LIDAR_FRAME, CAM_FRAME)
        T_cam_lidar = np.linalg.inv(T_lidar_cam)

    print("T_cam_lidar =\n", T_cam_lidar)

    
    # 2) Read rosbag and collect synchronized (rgb, lidar, cam_info) samples

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

    lidar_topic = "/livox/lidar"
    rgb_topic = "/zed/zed_node/rgb/image_rect_color/compressed"
    caminfo_topic = "/zed/zed_node/rgb/camera_info"

    if lidar_topic not in topic_types:
        raise RuntimeError(f"{lidar_topic} not found in bag")
    if rgb_topic not in topic_types:
        raise RuntimeError(f"{rgb_topic} not found in bag")
    if caminfo_topic not in topic_types:
        raise RuntimeError(f"{caminfo_topic} not found in bag")

    lidar_msg_type = get_message(topic_types[lidar_topic])
    rgb_msg_type = get_message(topic_types[rgb_topic])
    caminfo_msg_type = get_message(topic_types[caminfo_topic])

    last_lidar_msg = None
    last_caminfo_msg = None

    samples = []  # list of (timestamp, rgb_msg, lidar_msg, caminfo_msg)

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == lidar_topic:
            last_lidar_msg = deserialize_message(data, lidar_msg_type)

        elif topic == caminfo_topic:
            last_caminfo_msg = deserialize_message(data, caminfo_msg_type)

        elif topic == rgb_topic:
            if last_lidar_msg is None or last_caminfo_msg is None:
                continue
            rgb_msg = deserialize_message(data, rgb_msg_type)
            samples.append((t, rgb_msg, last_lidar_msg, last_caminfo_msg))

    if not samples:
        raise RuntimeError("No synchronized (RGB, LiDAR, CameraInfo) samples found.")

    print(f"Collected {len(samples)} synchronized samples.")

 
    # 3) Select sample by index (clamped to valid range)

    idx = max(0, min(TARGET_SAMPLE_INDEX, len(samples) - 1))
    stamp, rgb_msg, lidar_msg, caminfo_msg = samples[idx]
    print(f"Using sample index {idx} (timestamp={stamp}).")

  
    # 4) Convert to numpy: RGB image, camera matrix K, cloud_cam
   
    rgb_img = decode_compressed_image(rgb_msg)
    if rgb_img is None:
        raise RuntimeError("Decoded RGB image is None")

    # Camera intrinsics
    K = np.array(caminfo_msg.k, dtype=np.float32).reshape(3, 3)

    # LiDAR points in lidar frame
    cloud_lidar = pointcloud2_to_xyz_array(lidar_msg)
    print("LiDAR cloud (lidar frame):", cloud_lidar.shape)

    # Transform cloud to camera frame using T_cam_lidar
    N = cloud_lidar.shape[0]
    pts_lidar_h = np.hstack([cloud_lidar, np.ones((N, 1), dtype=np.float32)])  # (N, 4)
    pts_cam_h = (T_cam_lidar @ pts_lidar_h.T).T
    cloud_cam = pts_cam_h[:, :3]

    print("Cloud in camera frame:", cloud_cam.shape)

    # 5) Save outputs

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), rgb_img)
    np.save(out_cloud_path, cloud_cam)
    np.save(out_K_path, K)

    print(f"Saved RGB image to      : {out_img_path}")
    print(f"Saved camera cloud to   : {out_cloud_path}")
    print(f"Saved camera matrix K to: {out_K_path}")


if __name__ == "__main__":
    main()
