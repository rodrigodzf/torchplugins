/**
  \ingroup NNLib
  \file    PytorchFrontend
  \brief   This file contains the implementation for class PytorchFrontend.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <ATen/TensorIndexing.h>
#include <c10/core/TensorOptions.h>
#include <string>
#include <vector>
#include <array>
#include <memory>

#include <torch/script.h>


class PytorchFrontend
{

private:
    torch::jit::script::Module mModule;
    void printDebug();
    torch::TensorOptions options;

    std::vector<torch::jit::IValue> scaling_inputs;


public:
    std::vector<float> coefficients;

public:
    PytorchFrontend();
    ~PytorchFrontend();

    bool load(const std::string &filename);

    bool scale(
        const std::vector<float> &scaling
    )
    {
        scaling_inputs[2] = scaling[0]; // physical size
        scaling_inputs[3] = scaling[1]; // rho
        scaling_inputs[4] = scaling[2]; // E
        scaling_inputs[5] = scaling[3]; // alpha
        scaling_inputs[6] = scaling[4]; // beta
        scaling_inputs[7] = scaling[5]; // sample period

        auto ba = mModule.get_method("scale")(
            scaling_inputs
        ).toTensor();

        std::memcpy(
            coefficients.data(),
            ba.data_ptr(),
            ba.numel() * sizeof(float)
        );

        return true;
    };

    template <typename T>
    bool process(
        std::vector<T> &feature_vector,
        std::vector<T> &coords_vector
    )
    {
        c10::InferenceMode guard;

        torch::Tensor geometrical_features = torch::from_blob(
            feature_vector.data(),
            {1, feature_vector.size()},
            options
        );

        torch::Tensor coords = torch::from_blob(
            coords_vector.data(),
            {1, 2},
            options
        );

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(geometrical_features);
        inputs.push_back(coords);
        
        auto gains_theta = mModule.forward(inputs).toTuple()->elements();

        scaling_inputs[0] = gains_theta[0].toTensor();
        scaling_inputs[1] = gains_theta[1].toTensor();
        
        return true;
#if 0
        torch::Tensor feature_tensor = torch::from_blob(
            feature_vector.data(),
            {1, 32},
            options
        );

        // Get a pointer to the input tensors
        torch::Tensor material_tensor = torch::from_blob(
            materials_vector.data(),
            {1, 5},
            options
        );

        torch::Tensor coords_tensor = torch::from_blob(
            coords_vector.data(),
            {1, 2},
            options
        );

        torch::Tensor input_tensor = torch::cat(
            {feature_tensor, coords_tensor, material_tensor},
            1
        );
        
        // print the shape of the input tensor
        // std::cout << "Input tensor shape: " << input_tensor.sizes() << "\n";
        // std::cout << "Input tensor: " << input_tensor << "\n";

        // Execute the model and turn its output into a tensor.
        auto coefficientTensor = mModule.forward({input_tensor}).toTensor();
        
        // print the shape of the output tensor
        // std::cout << "Output tensor shape: " << coefficientTensor.sizes() << "\n";

        // print the first 10 coefficients along the last dimension
        // auto b = coefficientTensor.index(
        //     {0, 0, 0, torch::indexing::Slice(torch::indexing::None, 3)}
        // );
        // auto a = coefficientTensor.index(
        //     {0, 0, 0, torch::indexing::Slice(3, torch::indexing::None)}
        // );

        // std::cout << "b: " << b << "\n";
        // std::cout << "a: " << a << "\n";

        std::memcpy(
            coefficients.data(),
            coefficientTensor.data_ptr(),
            coefficientTensor.numel() * sizeof(float)
        );

        return true;
#endif
    }
};
