from pathlib import Path
import json
from typing import List, Dict

import cv2
import numpy as np

from aici.maps import OccupancyMap


# Same class → color style as office version
CLASS_COLORS = {
    "wc": (255, 0, 0),          # blue-ish
    "bathtub": (0, 255, 255),   # yellow-ish
    "sink": (0, 165, 255),      # orange-ish
    "chair": (0, 0, 255),       # red
    "couch": (0, 255, 0),       # green
    "table": (0, 255, 255),     # yellow
}


def draw_objects_with_ids_and_legend(
    occ: OccupancyMap,
    objects: List[Dict],
) -> np.ndarray:
    """
    Draw oriented boxes for each bathroom object, with:
      • object ID number (near box top-left)
      • legend text in bottom-left corner
    """

    # Convert PGM occupancy map → BGR image
    color_map = cv2.cvtColor(occ.map_img, cv2.COLOR_GRAY2BGR)

    legend_lines = []

    for obj_id, obj in enumerate(objects, start=1):
        cls = obj.get("class", "obj")

        fp = obj["footprint_cam"]
        x_map = fp["center_x"]   # Using approx camera frame ≈ map frame
        y_map = fp["center_z"]
        yaw = fp["yaw"]
        length = float(fp["length"])
        width = float(fp["width"])

        width = max(width, 0.20)  # prevent degenerate thin boxes

        # Convert world coordinates (meters) → pixel coordinates
        cx_px, cy_px = occ.world_to_pixel(x_map, y_map)
        cx_px = float(cx_px)
        cy_px = float(cy_px)

        # Convert length/width to pixel units
        half_len_px = (length / occ.resolution) / 2.0
        half_wid_px = (width  / occ.resolution) / 2.0

        # OpenCV rotated rectangle format
        rect = (
            (cx_px, cy_px),
            (2.0 * half_len_px, 2.0 * half_wid_px),
            -np.degrees(yaw),
        )
        box = cv2.boxPoints(rect).astype(int)

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        # Draw oriented box
        cv2.polylines(color_map, [box], isClosed=True, color=color, thickness=2)
        cv2.circle(color_map, (int(cx_px), int(cy_px)), 3, (0, 0, 0), -1)

        # Top-left corner = smallest X+Y
        tl_idx = np.argmin(box.sum(axis=1))
        tl = box[tl_idx]
        tl_x, tl_y = int(tl[0]), int(tl[1])

        # Draw the ID number
        cv2.putText(
            color_map,
            f"{obj_id}",
            (tl_x, tl_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        # Add to legend list
        legend_lines.append(f"{obj_id}: {cls}")

    # Draw legend bottom-left
    if legend_lines:
        h, w = color_map.shape[:2]
        line_height = 15
        total_height = line_height * len(legend_lines)
        y0 = max(20, h - total_height - 10)
        x0 = 10

        for i, text in enumerate(legend_lines):
            y = y0 + i * line_height
            cv2.putText(
                color_map,
                text,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return color_map


def main():
    ROOT = Path(__file__).resolve().parents[2]

    map_pgm = ROOT / "data" / "bathroom" / "room.pgm"
    map_yaml = ROOT / "data" / "bathroom" / "room.yaml"
    footprints_json = ROOT / "results" / "bathroom_object_footprints_cam.json"

    if not footprints_json.exists():
        raise FileNotFoundError(
            f"Footprints file not found: {footprints_json} "
            "(run `python3 -m aici.bathroom_fit_boxes` first)."
        )

    with open(footprints_json, "r") as f:
        objects = json.load(f)

    print(f"Loaded {len(objects)} bathroom objects from {footprints_json}")

    # Your OccupancyMap constructor takes (pgm_path, yaml_path)
    occ = OccupancyMap(str(map_pgm), str(map_yaml))

    # Draw boxes with IDs + legend
    color_map = draw_objects_with_ids_and_legend(occ, objects)

    # Save output
    out_png = ROOT / "results" / "bathroom_map_detections.png"
    cv2.imwrite(str(out_png), color_map)
    print(f"Saved bathroom map detections overlay to {out_png}")


if __name__ == "__main__":
    main()
