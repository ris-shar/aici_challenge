from pathlib import Path
import json
from typing import List, Dict

import cv2
import numpy as np

from aici.maps import OccupancyMap


# Simple color palette for classes
CLASS_COLORS = {
    "chair": (0, 0, 255),   # red
    "couch": (0, 255, 0),   # green
    "table": (0, 255, 255), # yellow
}


def draw_objects_with_ids_and_legend(
    occ: OccupancyMap,
    objects: List[Dict],
) -> np.ndarray:
    """
    Draw oriented boxes for each object, with:
      * an ID number near the top-left of the box
      * a legend text in the bottom-left corner of the map
    """

    # Start from grayscale map and convert to BGR
    color_map = cv2.cvtColor(occ.map_img, cv2.COLOR_GRAY2BGR)

    legend_lines = []

    for obj_id, obj in enumerate(objects, start=1):
        cls = obj.get("class", "obj")
        fp = obj["footprint_cam"]

        x_map = fp["center_x"]      # we approximate camera frame == map frame
        y_map = fp["center_z"]
        yaw = fp["yaw"]
        length = float(fp["length"])
        width = float(fp["width"])

        # Avoid degenerate tiny widths
        width = max(width, 0.20)

        # Convert world (meters) -> pixel coordinates
        cx_px, cy_px = occ.world_to_pixel(x_map, y_map)
        cx_px = float(cx_px)
        cy_px = float(cy_px)

        # Size in pixels
        half_len_px = (length / occ.resolution) / 2.0
        half_wid_px = (width / occ.resolution) / 2.0

        # OpenCV rotated rectangle uses degrees, image y-axis downward
        rect = (
            (cx_px, cy_px),
            (2.0 * half_len_px, 2.0 * half_wid_px),
            -np.degrees(yaw),
        )

        box = cv2.boxPoints(rect).astype(int)

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        # Draw oriented box and center
        cv2.polylines(color_map, [box], isClosed=True, color=color, thickness=2)
        cv2.circle(color_map, (int(cx_px), int(cy_px)), 3, (0, 0, 0), -1)

        # Choose a "top-left" corner in image coords (smallest x+y)
        tl_idx = np.argmin(box.sum(axis=1))
        tl = box[tl_idx]
        tl_x, tl_y = int(tl[0]), int(tl[1])

        # Draw ID number near that corner
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

        legend_lines.append(f"{obj_id}: {cls}")

    # Draw legend in bottom-left corner
    if legend_lines:
        # Start a bit above bottom, so all lines fit
        height, width = color_map.shape[:2]
        line_height = 15
        total_height = line_height * len(legend_lines)
        y0 = max(20, height - total_height - 10)
        x0 = 10

        for i, line in enumerate(legend_lines):
            y = y0 + i * line_height
            cv2.putText(
                color_map,
                line,
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

    map_pgm = ROOT / "data" / "office" / "room.pgm"
    map_yaml = ROOT / "data" / "office" / "room.yaml"
    footprints_json = ROOT / "results" / "office_object_footprints_cam.json"

    if not footprints_json.exists():
        raise FileNotFoundError(
            f"Footprints file not found: {footprints_json} "
            "(run `python3 -m aici.office_fit_boxes` first)."
        )

    with open(footprints_json, "r") as f:
        objects = json.load(f)

    print(f"Loaded {len(objects)} office objects from {footprints_json}")

    occ = OccupancyMap(str(map_pgm), str(map_yaml))

    color_map = draw_objects_with_ids_and_legend(occ, objects)

    out_png = ROOT / "results" / "office_map_detections.png"
    cv2.imwrite(str(out_png), color_map)
    print(f"Saved office map detections overlay to {out_png}")


if __name__ == "__main__":
    main()
