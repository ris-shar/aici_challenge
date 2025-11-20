import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# Classes relevant for Challenge 1
VALID_CLASSES = {
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",
    "dining table": "table",
    "table": "table",
    "toilet": "wc",
    "bathtub": "bathtub",
    "sink": "sink",  
}


def detect_objects_on_image(image_path: Path,
                            out_image: Path,
                            out_json_filtered: Path,
                            out_json_raw: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    print(f"Running YOLO on {image_path}")

    # Use a reasonably accurate YOLO model
    model = YOLO("yolov8m.pt")

    # Lower conf a bit to catch more bathroom objects
    results = model(img, conf=0.15)[0]

    raw_detections = []
    filtered_detections = []
    det_id = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = results.names[cls_id]
        conf = float(box.conf)

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        det = {
            "id": det_id,
            "class_raw": cls_name,
            "confidence": conf,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            },
        }
        raw_detections.append(det)

        if cls_name in VALID_CLASSES:
            mapped_class = VALID_CLASSES[cls_name]
            det_filtered = det.copy()
            det_filtered["class"] = mapped_class
            filtered_detections.append(det_filtered)

            # Draw them on the image
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"{mapped_class} {conf:.2f}",
                (int(x1), max(0, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        det_id += 1

    # Save outputs
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), img)

    out_json_filtered.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_filtered, "w") as f:
        json.dump(filtered_detections, f, indent=2)

    with open(out_json_raw, "w") as f:
        json.dump(raw_detections, f, indent=2)

    # Print a small summary
    classes_raw = {}
    for d in raw_detections:
        c = d["class_raw"]
        classes_raw[c] = classes_raw.get(c, 0) + 1

    print(f"Saved bathroom detection visualization to {out_image}")
    print(f"Saved {len(filtered_detections)} filtered detections to {out_json_filtered}")
    print(f"Saved {len(raw_detections)} raw detections to {out_json_raw}")
    print("Raw classes and counts:", classes_raw)


def main():
    ROOT = Path(__file__).resolve().parents[2]

    image_path = ROOT / "data" / "bathroom" / "bathroom_rgb_sample.png"
    out_image = ROOT / "results" / "bathroom_rgb_detections.png"
    out_json_filtered = ROOT / "results" / "bathroom_detections_2d.json"
    out_json_raw = ROOT / "results" / "bathroom_detections_2d_raw.json"

    print("Bathroom input image:", image_path)
    detect_objects_on_image(
        image_path,
        out_image,
        out_json_filtered,
        out_json_raw,
    )


if __name__ == "__main__":
    main()
