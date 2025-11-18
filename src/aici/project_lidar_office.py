from pathlib import Path
import json

import numpy as np
import cv2


def load_data(root: Path):
    img_path = root / "data" / "office" / "office_rgb_sample.png"
    cloud_path = root / "data" / "office" / "office_cloud_sample.npy"
    K_path = root / "data" / "office" / "office_cam_K.npy"
    det_json = root / "results" / "office_detections_2d.json"

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")

    cloud = np.load(cloud_path)  # shape (N, 3)
    K = np.load(K_path)          # shape (3, 3)

    detections = []
    if det_json.exists():
        with open(det_json, "r") as f:
            detections = json.load(f)

    print(f"Loaded image {img_path} with shape {img.shape}")
    print(f"Loaded cloud {cloud_path} with shape {cloud.shape}")
    print(f"Loaded K from {K_path}:\n{K}")
    print(f"Loaded {len(detections)} filtered detections")

    return img, cloud, K, detections, img_path


def project_points_to_image(cloud_cam: np.ndarray, K: np.ndarray, img_shape):
    """
    Simple pinhole projection: assumes cloud is already in camera coordinates.
    (Later we will insert the LiDAR->camera transform here.)
    """
    h, w = img_shape[:2]

    X = cloud_cam[:, 0]
    Y = cloud_cam[:, 1]
    Z = cloud_cam[:, 2]

    # Keep only points in front of camera
    valid = Z > 0.1
    X, Y, Z = X[valid], Y[valid], Z[valid]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # Keep points inside image bounds
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[mask].astype(np.int32)
    v = v[mask].astype(np.int32)

    print(f"Projected {len(u)} points into image")
    return u, v


def draw_overlay(img, u, v, detections):
    overlay = img.copy()

    # Draw LiDAR points
    for x, y in zip(u[::5], v[::5]):  # subsample for speed/clarity
        cv2.circle(overlay, (int(x), int(y)), 1, (0, 0, 255), -1)

    # Draw YOLO boxes for reference
    for det in detections:
        bbox = det["bbox"]
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        cls = det.get("class", det.get("class_raw", "obj"))
        conf = det.get("confidence", 0.0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"{cls} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return overlay


def main():
    ROOT = Path(__file__).resolve().parents[2]

    img, cloud, K, detections, img_path = load_data(ROOT)

    # *** IMPORTANT ASSUMPTION ***
    # We are treating LiDAR points as if they are already expressed
    # in the camera frame. This is *not* exact, but it lets us build
    # the projection + association pipeline first.
    cloud_cam = cloud.copy()

    u, v = project_points_to_image(cloud_cam, K, img.shape)
    overlay = draw_overlay(img, u, v, detections)

    out_path = ROOT / "results" / "office_rgb_lidar_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

    print(f"Saved LiDAR+YOLO overlay to {out_path}")


if __name__ == "__main__":
    main()
