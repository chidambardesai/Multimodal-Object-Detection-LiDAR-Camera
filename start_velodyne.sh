#!/bin/bash
sudo ip addr flush dev enp7s0
sudo ip addr add 192.168.1.101/24 dev enp7s0
sudo ip link set enp7s0 up

source /opt/ros/humble/setup.bash
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py
