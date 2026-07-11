#!/bin/bash

# abort if any command fails
set -e

echo "==== Building Blaze..."

mkdir -p blaze-install

if ! cd blaze; then
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze
    cd blaze
fi

cmake -DCMAKE_INSTALL_PREFIX=../blaze-install

make install

echo "==== Blaze installed"