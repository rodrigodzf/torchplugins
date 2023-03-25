#!/bin/bash

# parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            echo "Usage: ./build.sh [options]"
            echo "Options:"
            echo "  -h, --help            show this help message and exit"
            echo "  -c, --clean           clean build directory"
            echo "  -d, --debug           build in debug mode"
            echo "  -r, --release         build in release mode"
            echo "  -t, --test            build and run tests"
            echo "  -i, --install         install to system"
            exit 0
            ;;
        -c|--clean)
            rm -rf build
            rm -rf torchplugins
            rm -rf externals
            rm -rf tests
            rm -rf package-info.json
            exit 0
            ;;
        -d|--debug)
            build_type="Debug"
            shift
            ;;
        -r|--release)
            build_type="Release"
            shift
            ;;
        -t|--test)
            build_type="Release"
            test="ON"
            shift
            ;;
        -i|--install)
            build_type="Release"
            install="ON"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done


uname=$(uname)
arch=$(uname -m)

# macos (intel)

if [[ "$uname" == "Darwin" ]] && [[ "$arch" == "x86_64" ]]; then
    # download libtorch if it doesn't exist already
    if ! [[ -d "./third_party/libtorch" ]]; then
        cd ./third_party
        curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip -o libtorch.zip
        unzip libtorch.zip
        rm -rf libtorch.zip
        cd ../
    fi
fi

# macos (arm)

if [[ "$uname" == "Darwin" ]] && [[ "$arch" == "arm64" ]]; then
    # download libtorch if it doesn't exist already
    if ! [[ -d "./third_party/libtorch" ]]; then
        cd ./third_party
        curl -L https://anaconda.org/pytorch/pytorch/2.0.0/download/osx-arm64/pytorch-2.0.0-py3.9_0.tar.bz2 -o pytorch.tar.bz2
        mkdir pytorch
        tar -xvf pytorch.tar.bz2 -C pytorch
        cp -r pytorch/lib/python3.9/site-packages/torch libtorch
        rm -rf pytorch pytorch.tar.bz2
        cd ../
    fi
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=./third_party/libtorch
cmake --build build -j
cmake --install build
