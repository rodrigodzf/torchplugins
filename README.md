# Torchplugins

This repository contains a collection of plugins for loading pytorch models in Max/MSP, Unity and other engines/frameworks.

They can load different models, but they were developed to load the models used for [this project.](https://arxiv.org/abs/2210.15306)

## Cloning

```bash
git clone --recursive git@github.com:rodrigodzf/torchplugins.git
```

## Building

From the root directory, run:

```bash
mkdir build
cd build
cmake ..
make -j
```

The plugins will be available in the `externals` directory. It is recommended to copy the plugins to the `externals` directory of your Max installation e.g `~/Documents/Max 8/Packages/torchplugins/externals`.

## Usage

### Max/MSP

An exemplary patcher is available in the `extras` directory. The patcher loads a model and runs inference on a given image. The pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1dafuKfNBS-jLKJJZo5iVsO3d6K84xEmh?usp=share_link).

The path to the models must be specified in Max settings.

### Unity

### Related repositories

[nn-tilde](https://github.com/acids-ircam/nn_tilde) - A Max/MSP, PureData external for loading pytorch models.
