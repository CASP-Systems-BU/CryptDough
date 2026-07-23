#!/bin/bash

script_dir=$(dirname "$0")
log_dir="$script_dir/../../data/logs/table-2/cdough"
cd "$script_dir/ml"

mkdir -p "$log_dir"

# LAN experiments
./lan/ml_alexnet_3pc.sh >> "$log_dir/lan-ml_alexnet_3pc.log"
./lan/ml_vgg16_3pc.sh >> "$log_dir/lan-ml_vgg16_3pc.log"
./lan/ml_vgg16_imagenet_3pc.sh >> "$log_dir/lan-ml_vgg16_imagenet_3pc.log"

# WAN experiments
./wan/ml_alexnet_3pc.sh >> "$log_dir/wan-ml_alexnet_3pc.log"
./wan/ml_vgg16_3pc.sh >> "$log_dir/wan-ml_vgg16_3pc.log"
./wan/ml_vgg16_imagenet_3pc.sh >> "$log_dir/wan-ml_vgg16_imagenet_3pc.log"