# Neural Resonator for Max/MSP

This repository contains a collection of plugins for loading pytorch models in Max/MSP, Unity and other engines/frameworks.

They can load different models, but they were developed to load the models used for [this project.](https://arxiv.org/abs/2210.15306)

## Cloning

```bash
git clone --recursive git@github.com:rodrigodzf/torchplugins.git
```

## Building

From the root directory, run:

```bash
sh ./bin/build.sh -r
```

The plugins will be available in the `torchplugins` directory. It is recommended to copy the folder `torchplugins` to the Package directory of your Max installation e.g `~/Documents/Max 8/Packages/torchplugins`.

### Development

It is useful to link the folder `torchplugins` to the packages directory of your Max installation e.g `~/Documents/Max 8/Packages`. This way, you can edit the source code and the changes will be reflected in Max.

## Usage

It is required to install the fftw3 library. On macOS, it can be installed with homebrew:

```bash
brew install fftw
```

### Max/MSP

An exemplary patcher is available in the `extras` directory. The patcher loads a model and runs inference on a given image.

The path to the models must be specified in Max settings.

### Unity

### Related repositories

[nn-tilde](https://github.com/acids-ircam/nn_tilde) - A Max/MSP, PureData external for loading pytorch models.
