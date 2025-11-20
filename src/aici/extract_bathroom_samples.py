from pathlib import Path

import numpy as np
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import PointCloud2, CompressedImage, CameraInfo
from sensor_msgs_py import point_cloud2

from aici.tf_utils import load_tf_graph_from_bag


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    pts = []
    for p in point_cloud2.read_points(msg,
                                      field_names=("x", "y", "z"),
                                      skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def main():
    ROOT = Path(__file__).resolve().parents[2]
    bag_dir = ROOT / "data" / "bathroom" / "rosbag"

    out_img_path = ROOT / "data" / "bathroom" / "bathroom_rgb_sample.png"
    out_K_path = ROOT / "data" / "bathroom" / "bathroom_cam_K.npy"
    out_cloud_cam_path = ROOT / "data" / "bathroom" / "bathroom_cloud_cam.npy"

    rgb_topic = "/zed/zed_node/rgb/image_rect_color/compressed"
    caminfo_topic = "/zed/zed_node/rgb/camera_info"
    lidar_topic = "/livox/lidar"

    
    TARGET_RGB_INDEX = 658

    storage = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}
    msg_type = {name: get_message(t) for name, t in topic_types.items()}

    if rgb_topic not in topic_types or caminfo_topic not in topic_types or lidar_topic not in topic_types:
        raise RuntimeError("Missing RGB / CameraInfo / LiDAR topics in bag")

    rgb_msgs = []
    caminfo_msgs = []
    lidar_msgs = []

    # collect *all* messages in three aligned lists (same order as they appear)
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == rgb_topic:
            rgb_msgs.append(deserialize_message(data, msg_type[rgb_topic]))
        elif topic == caminfo_topic:
            caminfo_msgs.append(deserialize_message(data, msg_type[caminfo_topic]))
        elif topic == lidar_topic:
            lidar_msgs.append(deserialize_message(data, msg_type[lidar_topic]))

    n = min(len(rgb_msgs), len(caminfo_msgs), len(lidar_msgs))
    if n == 0:
        raise RuntimeError("No RGB/CamInfo/LiDAR triplets found in bathroom bag")

    idx = max(0, min(TARGET_RGB_INDEX, n - 1))
    print(f"Using bathroom frame index: {idx} (out of {n})")

    rgb_msg = rgb_msgs[idx]
    caminfo_msg = caminfo_msgs[idx]
    lidar_msg = lidar_msgs[idx]

    # RGB image
    np_arr = np.frombuffer(rgb_msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(str(out_img_path), img)
    print("Saved bathroom RGB:", out_img_path)

    # Camera intrinsics K
    K = np.array(caminfo_msg.k, dtype=np.float32).reshape(3, 3)
    np.save(out_K_path, K)
    print("Saved bathroom K:", out_K_path)

    # LiDAR -> camera frame
    cloud_lidar = pointcloud2_to_xyz_array(lidar_msg)
    print("Bathroom LiDAR cloud (lidar frame):", cloud_lidar.shape)

    tf_graph = load_tf_graph_from_bag(bag_dir)
    LIDAR_FRAME = "livox_frame"
    CAM_FRAME = "zed_left_camera_optical_frame"
    T_cam_lidar = tf_graph.get_transform(LIDAR_FRAME, CAM_FRAME)
    print("Bathroom T_cam_lidar =\n", T_cam_lidar)

    N = cloud_lidar.shape[0]
    pts_lidar_h = np.hstack([cloud_lidar, np.ones((N, 1), dtype=np.float32)])
    pts_cam_h = (T_cam_lidar @ pts_lidar_h.T).T
    cloud_cam = pts_cam_h[:, :3].astype(np.float32)

    np.save(out_cloud_cam_path, cloud_cam)
    print("Saved bathroom cloud_cam:", out_cloud_cam_path)


if __name__ == "__main__":
    main()
