import cv2
import yaml
import numpy as np

class OccupancyMap:
    def __init__(self, pgm_path, yaml_path):
        self.map_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise FileNotFoundError(f"Could not read map image: {pgm_path}")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.resolution = float(cfg["resolution"])   # meters/pixel
        self.origin = cfg["origin"]                  # [x0, y0, yaw]
        self.height, self.width = self.map_img.shape

    def world_to_pixel(self, x, y):
        x0, y0, _ = self.origin
        mx = (x - x0) / self.resolution
        my = (y - y0) / self.resolution
        u = int(mx)
        v = int(self.height - my)
        return u, v

def draw_oriented_box(color_img, occ: OccupancyMap,
                      x, y, yaw, length, width,
                      bgr=(0, 0, 255), thickness=2):
    cx, cy = occ.world_to_pixel(x, y)
    hl = (length / occ.resolution) / 2.0
    hw = (width  / occ.resolution) / 2.0

    rect = ((cx, cy), (2*hl, 2*hw), -np.degrees(yaw))
    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(color_img, [box], True, bgr, thickness)
    cv2.circle(color_img, (cx, cy), 3, (0, 255, 0), -1)
    return color_img
