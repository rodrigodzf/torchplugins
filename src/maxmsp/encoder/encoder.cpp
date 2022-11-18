#include "c74_min.h"

#include "model.h"
#include "log.h"

#ifndef VERSION
#define VERSION "0.0.0"
#endif

using namespace c74::min;

class Encoder : public object<Encoder> {
public:
    MIN_DESCRIPTION	{"Encode an image to a vector of floats."};
    MIN_TAGS		{"torch"};
    MIN_AUTHOR		{"Rodrigo Diaz"};
    MIN_RELATED		{"print"};

    Encoder(const atoms &args = {});
    ~Encoder();

	inlet<>		input	{ this, "(list) values to convolve" };
    inlet<>		input2	{ this, "(matrix) Input", "matrix" };

	outlet<>	output	{ this, "(list) result of convolution" };

    attribute<int> width
    { 
        this,
        "width",
        64,
        description {"Width of the input image."},
    };

    attribute<int> height
    { 
        this,
        "height",
        64,
        description {"Height of the input image."},
    };

    attribute<int> channels
    { 
        this,
        "channels",
        1,
        description {"Number of channels of the input image."},
    };

    attribute<int> batch_size
    { 
        this,
        "batch_size",
        1,
        description {"Number of images to encode."},
    };

    attribute<int> num_features
    { 
        this,
        "num_features",
        1000,
        description {"Number of features to encode."},
    };

    message<> list { this, "list", "Input to the convolution function.",
        MIN_FUNCTION {
            lock lock {m_mutex};

            std::vector<float> x = from_atoms<std::vector<float>>(args);
            if (m_output.size() != num_features)
            {
                m_output.resize(num_features);
            }

            auto tensor = Model::getInstance().toTensor(
                x.data(),
                batch_size,
                channels,
                height,
                width
            );
            
            // for images we need to repeat the tensor to match the number of channels
            tensor = tensor.repeat({1, 3, 1, 1});

            Model::getInstance().process(
                tensor,
                m_output.data()
            );

            lock.unlock();
            output.send(to_atoms(m_output));

            return {};
        }
    };

    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION
        {
            cout << "encoder - " << VERSION << " - 2022 - Rodrigo Diaz" << endl;
            return {};
        }
    };
private:
    std::mutex m_mutex;
    std::vector<float> m_output;
};

Encoder::Encoder(const atoms &args) {

    if (args.empty()) {
        error("Please provide a path to the model.");
        return;
    }

    auto modelPath = std::string(args[0]);

    // load model
    if (0 == Model::getInstance().loadModel(modelPath, "cpu"))
    {
        cout << "model with path: " << modelPath << " loaded" << endl;
    }
    else
    {
        cout << "model could not be loaded" << endl;
    }


}

Encoder::~Encoder() {}


MIN_EXTERNAL(Encoder);
