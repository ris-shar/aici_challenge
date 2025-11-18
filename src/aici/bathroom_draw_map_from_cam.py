from pathlib import Path
import json

import cv2

from aici.maps import OccupancyMap, draw_oriented_box


def main():
    ROOT = Path(__file__).resolve().parents[2]

    # Map files for bathroom
    map_pgm = ROOT / "data" / "bathroom" / "room.pgm"
    map_yaml = ROOT / "data" / "bathroom" / "room.yaml"

    # 3D footprints we computed in camera frame
    footprints_json = ROOT / "results" / "bathroom_object_footprints_cam.json"

    if not footprints_json.exists():
        raise FileNotFoundError(
            f"Footprints file not found: {footprints_json} "
            "(run aici.bathroom_fit_boxes first)"
        )

    with open(footprints_json, "r") as f:
        objects = json.load(f)

    print(f"Loaded {len(objects)} bathroom objects from {footprints_json}")

    # Load occupancy map
    occ = OccupancyMap(str(map_pgm), str(map_yaml))
    color_map = cv2.cvtColor(occ.map_img, cv2.COLOR_GRAY2BGR)

    # Approximation: camera frame == map frame
    for obj in objects:
        fp = obj["footprint_cam"]
        cls = obj["class"]

        x_cam = fp["center_x"]
        z_cam = fp["center_z"]
        yaw_cam = fp["yaw"]
        length = fp["length"]
        width = fp["width"]

        # Clamp width to avoid degenerate slivers
        width = max(width, 0.2)

        x_map = x_cam
        y_map = z_cam
        yaw_map = yaw_cam

        print(
            f"Drawing bathroom {cls}: "
            f"map_x={x_map:.2f}, map_y={y_map:.2f}, "
            f"yaw={yaw_map:.2f}, L={length:.2f}, W={width:.2f}"
        )

        # Simple color coding
        if cls == "wc":
            color = (255, 0, 0)      # blue-ish
        elif cls == "bathtub":
            color = (0, 255, 255)    # yellow-ish
        elif cls == "sink":
            color = (0, 165, 255)    # orange-ish
        else:
            color = (0, 255, 0)      # green for others

        color_map = draw_oriented_box(
            color_map,
            occ,
            x_map,
            y_map,
            yaw_map,
            length,
            width,
            bgr=color,
            thickness=2,
        )

    out_png = ROOT / "results" / "bathroom_map_detections.png"
    cv2.imwrite(str(out_png), color_map)
    print(f"Saved bathroom map detections overlay to {out_png}")


if __name__ == "__main__":
    main()
