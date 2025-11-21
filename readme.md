# AICI Challenge – Object Detection, Footprint Projection & World-Frame Point Cloud Mapping

**Author:** Rishav Sharma
**Challenges:**

* Challenge 1 – Object Projection Into Occupancy Grid Maps
* Challenge 2 – World-Frame Reconstruction From LiDAR + RGB
  **Sensors:** ZED Stereo RGB, Livox LiDAR, TF tree from MESA Robot (ROS 2 Humble)

---

# Overview

This repository contains a complete implementation of **Challenge 1** and **Challenge 2** from the AICI Computer Vision & Software Developer Coding Challenge.

It includes:

* Extraction of synchronized data from ROS2 bags
* YOLOv8 object detection
* LiDAR–camera fusion using TF
* Footprint estimation using PCA
* Projection onto occupancy grid maps
* World-frame LiDAR colorization
* Multi-frame concatenation into a global point cloud

The system supports both the **Office** and **Bathroom** surveys.

---

# Repository Structure

```
aici_challenge/
│
├── data/
│   ├── office/
│   └── bathroom/
│
├── results/
│
├── results_achieved/          # Final exported results for both challenges
│
├── src/
│   └── aici/
│       ├── detection.py
│       ├── extract_office_samples.py
│       ├── extract_bathroom_samples.py
│       ├── office_fit_boxes.py
│       ├── bathroom_fit_boxes.py
│       ├── office_draw_map_from_cam.py
│       ├── bathroom_draw_map_from_cam.py
│       ├── colorize_concat_office.py
│       ├── colorize_concat_bathroom_world.py
│       ├── colorize_concat_office_world.py
│       ├── maps.py
│       ├── tf_utils.py
│       └── project_lidar_office_tf.py
│
├── Dockerfile
├── docker-compose-ch1.yml
├── docker-compose-ch2.yml
└── README.md
```

---

# Challenge 1 – Pipeline Summary

Challenge 1 requires detecting objects and projecting their 3D footprints into the occupancy grid maps.

The full pipeline consists of:

## 1. Extract synchronized RGB + LiDAR + CameraInfo + TF

Scripts:

```
src/aici/extract_office_samples.py
src/aici/extract_bathroom_samples.py
```

Outputs (per environment):

* `*_rgb_sample.png`
* `*_cloud_sample.npy` or `*_cloud_cam.npy`
* `*_cam_K.npy`

---

## 2. YOLOv8 2D Object Detection

Script:

```
src/aici/detection.py
```

Outputs:

* `*_rgb_detections.png`
* `*_detections_2d.json`
* `*_detections_2d_raw.json`

---

## 3. Fit 3D Object Footprints in Camera Frame

Scripts:

```
src/aici/office_fit_boxes.py
src/aici/bathroom_fit_boxes.py
```

Footprint estimation uses:

* Projection of LiDAR into image frame
* Depth filtering with median Z
* PCA on (X,Z) ground-plane coordinates
* Robust quantile bounds to prevent elongated boxes

Output:

* `*_object_footprints_cam.json`

---

## 4. Project footprints into occupancy grid

Scripts:

```
src/aici/office_draw_map_from_cam.py
src/aici/bathroom_draw_map_from_cam.py
```

Outputs:

* `office_map_detections.png`
* `bathroom_map_detections.png`

Each map shows:

* Oriented bounding boxes
* Numeric labels per object
* A legend listing class names and IDs

---

# Challenge 2 – World-Frame LiDAR Colorization & Reconstruction

Challenge 2 requires reconstruction of the environment by converting per-frame LiDAR to world coordinates using robot odometry and TF, and coloring each point using synchronized RGB.

The pipeline outputs both a **local camera-frame cloud** and a **global world-frame cloud**.

## 1. Local Colorized Cloud (per sample)

Scripts:

```
src/aici/colorize_concat_office.py
src/aici/colorize_concat_bathroom.py   (optional)
```

Output:

* `office_colorized_cloud.ply`
* `bathroom_colorized_cloud.ply`

These clouds represent LiDAR points colored by their corresponding RGB pixels but remain in the camera coordinate frame.

---

## 2. World-Frame Reconstruction (Global Map)

Scripts:

```
src/aici/colorize_concat_office_world.py
src/aici/colorize_concat_bathroom_world.py
```

Pipeline:

* Load TF transforms
* Transform LiDAR → camera
* Transform camera → base_link
* Transform base_link → odom (using TF)
* Concatenate multiple frames
* Colorize each point using RGB
* Export PLY

Outputs:

* `office_colorized_cloud_world.ply`
* `bathroom_colorized_cloud_world.ply`

These represent a colored 3D map of the full walk-around trajectory.

---

# Running With Docker

## Challenge 1 and 2

```bash
docker compose -f docker-compose-ch1.yml up
```
(If the image index has to be changed it can be done fin the extract_bathroom_samples.py and extract_office_samples.py)
This automatically executes all Challenge 1 adn 2 steps and writes output to:

```
results/
```

---

```
For now docker_compose_ch1.yml act as a unified compose file for challenge 1 and 2 but current working on challenge 3 will ensure different compose files for each tasks as well.

```


Place rosbags as:

```
data/office/rosbag/rosbag2_...db3
data/bathroom/rosbag/rosbag2_...db3
```

---

# Running Without Docker (Local Execution)

## Requirements

| Dependency         | Version    |
| ------------------ | ---------- |
| Python             | 3.10.12    |
| ROS 2              | Humble     |
| ultralytics        | ≥ 8.3      |
| OpenCV             | 4.11.0     |
| NumPy              | ≥ 1.23     |
| rosbag2_py         | ROS Humble |
| sensor_msgs_py     | ROS Humble |
| tf2_msgs / TF tree | ROS Humble |

Installation:

```bash
pip install ultralytics opencv-python numpy open3d matplotlib
source /opt/ros/humble/setup.bash
cd aici_challenge
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

---

# Manual Execution

## Challenge 1 – Office

```bash
python3 -m aici.extract_office_samples
python3 -m aici.detection
python3 -m aici.office_fit_boxes
python3 -m aici.office_draw_map_from_cam
```

## Challenge 1 – Bathroom

```bash
python3 -m aici.extract_bathroom_samples
python3 -m aici.bathroom_detection
python3 -m aici.bathroom_fit_boxes
python3 -m aici.bathroom_draw_map_from_cam
```

---

# Challenge 2 – World Frame Mapping

## Office World Map

```bash
python3 -m aici.colorize_concat_office_world
```

## Bathroom World Map

```bash
python3 -m aici.colorize_concat_bathroom_world
```

Visualize:

```bash
python3 - << 'EOF'
import open3d as o3d
pc = o3d.io.read_point_cloud("results/office_colorized_cloud_world.ply")
o3d.visualization.draw_geometries([pc])
EOF
```

---

# Notes and Limitations

* Occupancy maps may not perfectly align with TF-based reconstructions due to differences between SLAM and ground truth.
* YOLO detections vary across frames; some furniture may not be detected in every viewpoint.
* Rosbags are not included; users must place their own according to directory structure.

---

# Final Outputs

Outputs per survey include:

| File Type                      | Description                           |
| ------------------------------ | ------------------------------------- |
| `*_rgb_detections.png`         | 2D YOLO detections                    |
| `*_detections_2d.json`         | Filtered 2D detections                |
| `*_object_footprints_cam.json` | Estimated 3D footprints               |
| `*_map_detections.png`         | Final bounding boxes on occupancy map |
| `*_colorized_cloud_world.ply`  | World-frame colored LiDAR map         |

All final results used for submission are found in:

```
results_achieved/
```

---

# End
