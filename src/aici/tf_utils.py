import numpy as np
from pathlib import Path

import rosbag2_py
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _quat_to_rot(qx, qy, qz, qw):
    """Convert quaternion to a 3x3 rotation matrix."""
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    wx = qw*qx
    wy = qw*qy
    wz = qw*qz

    R = np.array([
        [1 - 2*(yy+zz),   2*(xy - wz),   2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),   1 - 2*(xx+yy)]
    ])
    return R


class TFGraph:
    """Simple TF graph to compute transforms between frames via BFS."""

    def __init__(self):
        # adjacency list: frame -> list of (neighbor_frame, T_neighbor_frame_from_frame)
        self.adj = {}

    def add_transform(self, parent, child, T_parent_child):
        # p_child = T_parent_child * p_parent
        self.adj.setdefault(parent, []).append((child, T_parent_child))

        # also store inverse: p_parent = T_child_parent * p_child
        T_child_parent = np.linalg.inv(T_parent_child)
        self.adj.setdefault(child, []).append((parent, T_child_parent))

    def frames(self):
        return list(self.adj.keys())

    def get_transform(self, source, target):
        """
        Return 4x4 matrix T_target_source such that:
            p_target = T_target_source @ p_source_h
        """
        if source == target:
            return np.eye(4)

        from collections import deque
        q = deque()
        q.append((source, np.eye(4)))
        visited = {source}

        while q:
            frame, T_acc = q.popleft()
            if frame == target:
                return T_acc

            for neigh, T_neigh_frame in self.adj.get(frame, []):
                if neigh in visited:
                    continue
                # p_neigh = T_neigh_frame @ p_frame
                # accumulated T from source to neigh:
                T_src_neigh = T_neigh_frame @ T_acc
                q.append((neigh, T_src_neigh))
                visited.add(neigh)

        raise ValueError(f"No TF path found from '{source}' to '{target}'")


def load_tf_graph_from_bag(bag_dir: Path) -> TFGraph:
    """Read all /tf_static messages from a rosbag2 folder and build a TFGraph."""
    bag_dir = Path(bag_dir)

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

    # topic -> type map
    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}

    if "/tf_static" not in topic_types:
        raise RuntimeError(f"No /tf_static topic found in bag: {bag_dir}")

    tf_msg_type = get_message(topic_types["/tf_static"])

    graph = TFGraph()

    # Manual filter: iterate everything, only handle /tf_static
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != "/tf_static":
            continue

        msg = deserialize_message(data, tf_msg_type)
        assert isinstance(msg, TFMessage)

        for transform in msg.transforms:
            assert isinstance(transform, TransformStamped)
            parent = transform.header.frame_id
            child = transform.child_frame_id

            t_ = transform.transform.translation
            q_ = transform.transform.rotation

            R = _quat_to_rot(q_.x, q_.y, q_.z, q_.w)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [t_.x, t_.y, t_.z]

            graph.add_transform(parent, child, T)

    return graph


def debug_print_frames(bag_dir: Path):
    """Debug helper: print all frames from /tf_static in this bag."""
    graph = load_tf_graph_from_bag(bag_dir)
    print("=== TF Frames in bag ===")
    for f in sorted(graph.frames()):
        print(" -", f)
