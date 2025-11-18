import json
from pathlib import Path

import numpy as np
import cv2


def project_points_cam_to_image(points_cam: np.ndarray, K: np.ndarray):
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    valid = Z > 0.1
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    return u.astype(np.int32), v.astype(np.int32), valid


def fit_footprint_from_points(points_cam: np.ndarray):
    if points_cam.shape[0] < 10:
        return None

    pts = points_cam[:, [0, 2]]
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid

    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    R2 = Vt
    proj = pts_c @ R2.T
    min0, max0 = proj[:, 0].min(), proj[:, 0].max()
    min1, max1 = proj[:, 1].min(), proj[:, 1].max()

    length = float(max0 - min0)
    width = float(max1 - min1)

    if width > length:
        length, width = width, length
        yaw = np.arctan2(R2[1, 1], R2[1, 0])
    else:
        yaw = np.arctan2(R2[0, 1], R2[0, 0])

    center_x, center_z = float(centroid[0]), float(centroid[1])
    return center_x, center_z, float(yaw), length, width


def main():
    ROOT = Path(__file__).resolve().parents[2]

    img_path = ROOT / "data" / "bathroom" / "bathroom_rgb_sample.png"
    K_path = ROOT / "data" / "bathroom" / "bathroom_cam_K.npy"
    cloud_path = ROOT / "data" / "bathroom" / "bathroom_cloud_sample.npy"
    det_json_path = ROOT / "results" / "bathroom_detections_2d.json"
    out_json_path = ROOT / "results" / "bathroom_object_footprints_cam.json"

    K = np.load(K_path)
    cloud_cam = np.load(cloud_path)
    print("Loaded bathroom K:\n", K)
    print("Bathroom cloud (cam frame):", cloud_cam.shape)

    with open(det_json_path, "r") as f:
        detections = json.load(f)
    print(f"Loaded {len(detections)} bathroom detections")

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read bathroom image at {img_path}")
    H, W, _ = img.shape

    u_all, v_all, valid_mask = project_points_cam_to_image(cloud_cam, K)
    cloud_valid = cloud_cam[valid_mask]

    footprints = []

    for det in detections:
        bbox = det["bbox"]
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        x1i = max(0, min(W - 1, int(x1)))
        y1i = max(0, min(H - 1, int(y1)))
        x2i = max(0, min(W - 1, int(x2)))
        y2i = max(0, min(H - 1, int(y2)))

        if x2i <= x1i or y2i <= y1i:
            continue

        inside = (u_all >= x1i) & (u_all <= x2i) & (v_all >= y1i) & (v_all <= y2i)
        pts_in_box = cloud_valid[inside]
        print(f"[Bathroom] Det {det['id']} ({det['class']}), points in box: {pts_in_box.shape[0]}")

        if pts_in_box.shape[0] < 15:
            continue

        Z = pts_in_box[:, 2]
        med_z = np.median(Z)
        depth_window = max(0.5, 0.3 * med_z)
        depth_mask = np.abs(Z - med_z) < depth_window
        pts_filtered = pts_in_box[depth_mask]

        print(f"[Bathroom] Det {det['id']} depth-filtered points: {pts_filtered.shape[0]}")

        if pts_filtered.shape[0] < 8:
            continue

        fp = fit_footprint_from_points(pts_filtered)
        if fp is None:
            continue

        center_x, center_z, yaw, length, width = fp

        out_det = det.copy()
        out_det["bbox_2d"] = out_det.pop("bbox")
        out_det["footprint_cam"] = {
            "center_x": center_x,
            "center_z": center_z,
            "yaw": yaw,
            "length": length,
            "width": width,
        }
        footprints.append(out_det)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(footprints, f, indent=2)

    print(f"Saved {len(footprints)} bathroom footprints to {out_json_path}")


if __name__ == "__main__":
    main()
