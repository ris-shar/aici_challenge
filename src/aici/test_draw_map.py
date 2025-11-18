from aici.maps import OccupancyMap, draw_oriented_box
import cv2
import math
from pathlib import Path

# Compute project root dynamically
ROOT = Path(__file__).resolve().parents[2]

def main():
    MAP_PGM = ROOT / "data/office/room.pgm"
    MAP_YAML = ROOT / "data/office/room.yaml"
    OUT_PNG = ROOT / "results/office_map_dummy.png"

    print("MAP_PGM =", MAP_PGM)
    print("MAP_YAML =", MAP_YAML)
    print("OUT_PNG =", OUT_PNG)

    occ = OccupancyMap(str(MAP_PGM), str(MAP_YAML))

    # Convert grayscale map to color
    color_map = cv2.cvtColor(occ.map_img, cv2.COLOR_GRAY2BGR)

    # Dummy objects
    objects = [
        {"x": -2.0, "y": -5.0, "yaw": 0.0,             "L": 0.6, "W": 0.6},
        {"x": -1.0, "y": -6.0, "yaw": math.radians(45), "L": 1.2, "W": 0.8},
    ]

    for o in objects:
        color_map = draw_oriented_box(
            color_map, occ,
            o["x"], o["y"], o["yaw"],
            o["L"], o["W"]
        )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), color_map)
    print(f"Saved {OUT_PNG}")

if __name__ == "__main__":
    main()

