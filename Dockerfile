# Base image with ROS 2 Humble on Ubuntu 22.04
FROM ros:humble-ros-base

# Avoid interactive apt dialogs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies: Python, OpenCV, ROS bag bindings, message types
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-ros2bag \
    ros-humble-sensor-msgs \
    ros-humble-tf2-msgs \
    ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies 
RUN pip3 install --no-cache-dir \
    "ultralytics==8.3.134" \
    numpy \
    matplotlib

# Create workspace directory
WORKDIR /ws

# Copy everything into the image (code, data, results structure)
COPY . /ws

# Make sure Python can see your src/ folder as a package root
ENV PYTHONPATH=/ws/src:${PYTHONPATH}

# Default: shell inside container
CMD ["bash"]

