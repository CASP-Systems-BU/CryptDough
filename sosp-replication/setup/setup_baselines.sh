#!/bin/bash

setup_dir=$(dirname "$0")

$setup_dir/setup_orq.sh node0 node1 node2 node3
$setup_dir/setup_tva.sh node0 node1 node2 node3
$setup_dir/setup_pigeon.sh node0 node1 node2 node3
$setup_dir/setup_piranha.sh node0 node1 node2 node3
$setup_dir/setup_mpspdz.sh node0 node1 node2 node3