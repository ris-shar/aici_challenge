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
    for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def main():
    ROOT = Path(__file__).resolve().parents[2]
    bag_dir = ROOT / "data" / "bathroom" / "rosbag"

    out_img_path = ROOT / "data" / "bathroom" / "bathroom_rgb_sample.png"
    out_K_path = ROOT / "data" / "bathroom" / "bathroom_cam_K.npy"
    out_cloud_cam_path = ROOT / "data" / "bathroom" / "bathroom_cloud_sample.npy"
    out_cloud_cam_path2 = ROOT / "data" / "bathroom" / "bathroom_cloud_cam.npy"

    rgb_topic = "/zed/zed_node/rgb/image_rect_color/compressed"
    caminfo_topic = "/zed/zed_node/rgb/camera_info"
    lidar_topic = "/livox/lidar"

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
    msg_type = {name: get_message(t) for name, t in topic_types.items()}

    if rgb_topic not in topic_types or caminfo_topic not in topic_types or lidar_topic not in topic_types:
        raise RuntimeError("One of RGB / CameraInfo / LiDAR topics not in bag")

    rgb_msg = caminfo_msg = lidar_msg = None

    while reader.has_next() and (rgb_msg is None or caminfo_msg is None or lidar_msg is None):
        topic, data, t = reader.read_next()
        if topic == rgb_topic and rgb_msg is None:
            rgb_msg = deserialize_message(data, msg_type[rgb_topic])
        elif topic == caminfo_topic and caminfo_msg is None:
            caminfo_msg = deserialize_message(data, msg_type[caminfo_topic])
        elif topic == lidar_topic and lidar_msg is None:
            lidar_msg = deserialize_message(data, msg_type[lidar_topic])

    if rgb_msg is None or caminfo_msg is None or lidar_msg is None:
        raise RuntimeError("Could not read RGB / CameraInfo / LiDAR")

    # RGB
    np_arr = np.frombuffer(rgb_msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(str(out_img_path), img)
    print("Saved bathroom RGB to:", out_img_path)

    # K
    K = np.array(caminfo_msg.k, dtype=np.float32).reshape(3, 3)
    np.save(out_K_path, K)
    print("Saved bathroom K to:", out_K_path)

    # LiDAR → camera frame
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
    np.save(out_cloud_cam_path2, cloud_cam)
    print("Saved bathroom cloud in camera frame:", out_cloud_cam_path)


if __name__ == "__main__":
    main()
