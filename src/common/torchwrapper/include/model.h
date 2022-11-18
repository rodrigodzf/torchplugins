#pragma once

#include <torch/script.h>
#include <vector>
#include <cstdio>
#include <cmath>

class Model
{

private:
    Model() {}
public:
    Model(Model const &) = delete;
    void operator=(Model const &) = delete;
private:
    torch::jit::Module mNetwork;

public:
    static Model &getInstance()
    {
        static Model instance; // Guaranteed to be destroyed.
        return instance;
    }

    int loadModel(
        const std::string &modelPath,
        const std::string &deviceString
    );

    torch::Tensor toTensor(
        float *input,
        const int batchSize,
        const int channels,
        const int height,
        const int width
    );

    void process(
        torch::Tensor &inputTensor,
        float* output
    );
};
