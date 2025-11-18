import json
from pathlib import Path

import numpy as np


MIN_POINTS_IN_BOX = 80           # need at least this many LiDAR points
DEPTH_WINDOW_RATIO = 0.3         # keep points within ±30% of median depth
LONG_AXIS_Q = 95.0               # % of points covered along long axis
SHORT_AXIS_Q = 95.0              # % along short axis
MIN_LENGTH = 0.4                 # m, avoid degenerate tiny boxes
MIN_WIDTH = 0.3                  # m


def project_points_cam_to_image(points_cam: np.ndarray, K: np.ndarray):
    """
    points_cam: (N, 3), in camera frame  (X=right, Y=down, Z=forward)
    returns u, v, Z, valid_mask
    """
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = Z > 0.1  # only in front of camera
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    return u, v, Z, valid


def fit_oriented_box(points_cam: np.ndarray):
    """
    Fit an oriented 2D box to points in *camera* frame.
    Uses PCA + quantiles so the box is not over-stretched by outliers.
    Returns (center_x, center_z, yaw, length, width) or None.
    """
    if points_cam.shape[0] < MIN_POINTS_IN_BOX:
        return None

    # 1) depth filter around median Z
    Z = points_cam[:, 2]
    z_med = np.median(Z)
    z_window = DEPTH_WINDOW_RATIO * z_med
    depth_mask = (Z > z_med - z_window) & (Z < z_med + z_window)
    pts = points_cam[depth_mask]

    if pts.shape[0] < MIN_POINTS_IN_BOX:
        return None

    # 2) ground-plane coordinates (X,Z)
    pts2d = pts[:, [0, 2]]  # (M,2)
    center = pts2d.mean(axis=0)

    # 3) PCA on centered points
    pts_centered = pts2d - center
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    idx_long = np.argmax(eigvals)
    long_axis = eigvecs[:, idx_long]
    short_axis = eigvecs[:, 1 - idx_long]

    # 4) coordinates in PCA basis
    coords_long = pts_centered @ long_axis
    coords_short = pts_centered @ short_axis

    half_len = np.percentile(np.abs(coords_long), LONG_AXIS_Q)
    half_wid = np.percentile(np.abs(coords_short), SHORT_AXIS_Q)

    length = max(2.0 * half_len, MIN_LENGTH)
    width = max(2.0 * half_wid, MIN_WIDTH)

    # yaw in camera frame (X,Z plane)
    yaw = np.arctan2(long_axis[1], long_axis[0])

    center_x, center_z = center[0], center[1]
    return center_x, center_z, yaw, length, width


def main():
    ROOT = Path(__file__).resolve().parents[2]

    cloud_path = ROOT / "data" / "office" / "office_cloud_sample.npy"
    K_path = ROOT / "data" / "office" / "office_cam_K.npy"
    det_json_path = ROOT / "results" / "office_detections_2d.json"
    out_json_path = ROOT / "results" / "office_object_footprints_cam.json"

    cloud_cam = np.load(cloud_path)          # (N,3) in camera frame
    K = np.load(K_path)
    print("Loaded camera-frame cloud:", cloud_cam.shape)
    print("Loaded K:\n", K)

    with open(det_json_path, "r") as f:
        detections = json.load(f)
    print(f"Loaded {len(detections)} 2D detections")

    # project all LiDAR points once
    u_all, v_all, Z_all, valid_mask = project_points_cam_to_image(cloud_cam, K)
    pts_valid = cloud_cam[valid_mask]

    objects_with_fp = []

    for det in detections:
        cls_name = det.get("class", det.get("class_raw", ""))

        # only keep relevant furniture classes
        if cls_name not in {"chair", "couch", "table", "wc", "bathtub"}:
            continue

        # support both "bbox_2d" and old "bbox" key
        if "bbox_2d" in det:
            bbox = det["bbox_2d"]
        else:
            bbox = det["bbox"]

        x1, y1 = bbox["x1"], bbox["y1"]
        x2, y2 = bbox["x2"], bbox["y2"]

        mask_box = (
            (u_all >= x1) & (u_all <= x2) &
            (v_all >= y1) & (v_all <= y2)
        )
        pts_in_box = pts_valid[mask_box]
        print(f"Det {det['id']} ({cls_name}) initial points in box: {pts_in_box.shape[0]}")

        fp = fit_oriented_box(pts_in_box)
        if fp is None:
            print(f"[SKIP] not enough clean LiDAR points for det {det['id']}")
            continue

        cx, cz, yaw, length, width = fp
        print(
            f"Det {det['id']} footprint: "
            f"center=({cx:.2f},{cz:.2f}), yaw={yaw:.2f} rad, "
            f"LxW={length:.2f}x{width:.2f} m"
        )

        det_out = dict(det)
        det_out["footprint_cam"] = {
            "center_x": float(cx),
            "center_z": float(cz),
            "yaw": float(yaw),
            "length": float(length),
            "width": float(width),
        }
        objects_with_fp.append(det_out)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(objects_with_fp, f, indent=2)

    print(f"Saved {len(objects_with_fp)} footprints to {out_json_path}")


if __name__ == "__main__":
    main()
