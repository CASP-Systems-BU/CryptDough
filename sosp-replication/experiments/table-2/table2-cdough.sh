#!/bin/bash

script_dir=$(dirname "$0")
log_dir="$script_dir/../../data/logs/table-2/cdough"

mkdir -p "$log_dir"

# LAN experiments
$script_dir/ml/lan/ml_alexnet_3pc.sh >> "$log_dir/lan-3pc-alexnet.log"
$script_dir/ml/lan/ml_vgg16_3pc.sh >> "$log_dir/lan-3pc-vgg16.log"
$script_dir/ml/lan/ml_vgg16_imagenet_3pc.sh >> "$log_dir/lan-3pc-vgg16_imagenet.log"

# TODO: turn on WAN

# WAN experiments
$script_dir/ml/wan/ml_alexnet_3pc.sh >> "$log_dir/wan-3pc-alexnet.log"
$script_dir/ml/wan/ml_vgg16_3pc.sh >> "$log_dir/wan-3pc-vgg16.log"
$script_dir/ml/wan/ml_vgg16_imagenet_3pc.sh >> "$log_dir/wan-3pc-vgg16_imagenet.log"