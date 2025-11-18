from pathlib import Path
import json

import cv2
import numpy as np
import yaml


def load_map(root: Path):
    """Load occupancy grid (PGM) and YAML meta (resolution, origin)."""
    yaml_path = root / "data" / "office" / "room.yaml"
    pgm_path = root / "data" / "office" / "room.pgm"

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    resolution = float(cfg["resolution"])          # m / pixel
    origin = cfg["origin"]                         # [x, y, yaw]
    origin_x, origin_y = float(origin[0]), float(origin[1])

    img_gray = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read map image: {pgm_path}")

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    H, W = img_color.shape[:2]
    print(f"Loaded map {pgm_path} with size (HxW)=({H},{W}), "
          f"resolution={resolution}, origin=({origin_x},{origin_y})")

    return img_color, resolution, origin_x, origin_y


def world_to_map(x: float, y: float,
                 resolution: float,
                 origin_x: float,
                 origin_y: float,
                 img_shape):
    """
    Convert world coordinates (meters) to pixel coordinates in the PGM.

    Standard ROS nav convention:
      - origin (origin_x, origin_y) in meters corresponds to pixel (0, height-1)
      - x increases to the right, y increases up (in meters)
      - image (0,0) is top-left, row index downward

    So:
      mx = (x - origin_x) / res
      my = height - (y - origin_y) / res
    """
    H, W = img_shape[:2]

    mx = int((x - origin_x) / resolution)
    my = int(H - (y - origin_y) / resolution)

    return mx, my


def draw_oriented_box(img,
                      center_x: float,
                      center_y: float,
                      length: float,
                      width: float,
                      yaw: float,
                      resolution: float,
                      origin_x: float,
                      origin_y: float,
                      color=(0, 0, 255),
                      thickness=2):
    """
    Draw an oriented rectangle in *world* coordinates onto the map image.

    center_x, center_y: in meters (map/world frame)
    length, width: in meters
    yaw: orientation in radians in the XY plane
    """
    # Half extents
    hl = length / 2.0
    hw = width / 2.0

    # Direction vectors in world coordinates
    # Axis along the length
    dx = hl * np.cos(yaw)
    dy = hl * np.sin(yaw)

    # Axis along the width (perpendicular to length)
    wx = -hw * np.sin(yaw)
    wy = hw * np.cos(yaw)

    # Four corners in world frame
    corners_world = np.array([
        [center_x + dx + wx, center_y + dy + wy],
        [center_x + dx - wx, center_y + dy - wy],
        [center_x - dx - wx, center_y - dy - wy],
        [center_x - dx + wx, center_y - dy + wy],
    ], dtype=np.float32)

    # Convert to pixel coords
    pts_img = []
    for (xw, yw) in corners_world:
        mx, my = world_to_map(
            xw, yw,
            resolution, origin_x, origin_y,
            img.shape
        )
        pts_img.append([mx, my])

    pts_img = np.array(pts_img, dtype=np.int32).reshape((-1, 1, 2))

    cv2.polylines(img, [pts_img], isClosed=True, color=color, thickness=thickness)

    # Draw center as a small dot
    cx_pix, cy_pix = world_to_map(
        center_x, center_y,
        resolution, origin_x, origin_y,
        img.shape
    )
    cv2.circle(img, (cx_pix, cy_pix), 3, color, -1)


def main():
    ROOT = Path(__file__).resolve().parents[2]

    # 1. Load map
    map_img, resolution, origin_x, origin_y = load_map(ROOT)

    # 2. Load object footprints
    footprints_path = ROOT / "results" / "office_object_footprints_cam.json"
    with open(footprints_path, "r") as f:
        objs = json.load(f)

    print(f"Loaded {len(objs)} objects from {footprints_path}")

    # 3. Draw each object
    for obj in objs:
        # Support both flat format and old nested "footprint_cam" format
        if "footprint_cam" in obj:
            fp = obj["footprint_cam"]
        else:
            fp = obj

        cls_name = obj.get("class", "object")

        cx = float(fp["center_x"])
        cy = float(fp["center_z"])   # we stored Z as forward distance; use as Y in map
        yaw = float(fp["yaw"])
        length = float(fp["length"])
        width = float(fp["width"])

        print(f"Drawing {cls_name}: "
              f"map_x={cx:.2f}, map_y={cy:.2f}, yaw={yaw:.2f}, "
              f"L={length:.2f}, W={width:.2f}")

        # Choose color by class
        if "couch" in cls_name.lower():
            color = (0, 255, 255)   # yellow-ish
        elif "chair" in cls_name.lower():
            color = (0, 0, 255)     # red
        elif "table" in cls_name.lower():
            color = (0, 255, 0)     # green
        else:
            color = (255, 0, 0)     # blue

        draw_oriented_box(
            map_img,
            center_x=cx,
            center_y=cy,
            length=length,
            width=width,
            yaw=yaw,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            color=color,
            thickness=2,
        )

    # 4. Save result
    out_path = ROOT / "results" / "office_map_detections.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), map_img)
    print("Saved map detections overlay to", out_path)


if __name__ == "__main__":
    main()
