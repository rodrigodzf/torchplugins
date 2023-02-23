#include "model.h"
#include "log.h"

int Model::loadModel(
    const std::string &modelPath,
    const std::string &deviceString
)
{   
    auto device = torch::Device(deviceString);
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        mNetwork = torch::jit::load(modelPath, device);
        mNetwork.eval();
    }
    catch (const c10::Error &e)
    {
        // Return a value of -1 if the model fails to load
        LOG_MSG("Error loading model: " << e.what())
        return -1;
    }

    LOG_MSG("Model loaded successfully");

    // Return a value of 0 if the model loads successfully
    return 0;
}

torch::Tensor Model::toTensor(
    float *input,
    const int batchSize,
    const int channels,
    const int height,
    const int width
)
{
   // Create a vector of inputs.
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    auto tensor = torch::from_blob(
        input, 
        {batchSize, channels, height, width}, 
        options
    );

    return tensor;
}

void Model::process(
    torch::Tensor &inputTensor,
    float* output
)
{
    c10::InferenceMode guard;

    try
    {
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);

        // Execute the model and turn its output into a tensor.
        auto outputTensor = mNetwork.forward(inputs).toTensor();

        // Copy the tensor data to the output array
        memcpy(output, outputTensor.data_ptr(), outputTensor.numel() * sizeof(float));
    }
    catch (const c10::Error &e)
    {
        // Return a value of -1 if the model fails to load
        LOG_MSG("Error processing model: " << e.what());
        std::exit(-1);
    }
}

std::vector<std::string> Model::getMethods()
{
    std::vector<std::string> methods;
    for (const auto &method : mNetwork.get_methods())
    {
        methods.push_back(method.name());
    }
    return methods;
}

void Model::callMethod(
    const std::string &methodName,
    float param,
    float* output
)
{
    c10::InferenceMode guard;

    try
    {
        mNetwork.get_method(methodName)({param});
    }
    catch (const c10::Error &e)
    {
        // Return a value of -1 if the model fails to load
        LOG_MSG("Error processing model: " << e.what());
        std::exit(-1);
    }
}