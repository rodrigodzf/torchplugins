#include "c74_min.h"

#include "model.h"
#include "log.h"

#ifndef VERSION
#define VERSION "0.0.0"
#endif

using namespace c74::min;

class FC : public object<FC> {
public:
    MIN_DESCRIPTION	{"Encode an image to a vector of floats."};
    MIN_TAGS		{"torch"};
    MIN_AUTHOR		{"Rodrigo Diaz"};
    MIN_RELATED		{"print"};

    FC(const atoms &args = {});
    ~FC();

	inlet<>		input	{ this, "(list) flattened input" };
	outlet<>	output	{ this, "(list) features" };

    attribute<int> out_features
    { 
        this,
        "out_features",
        32 * 2 * 6,
        description {"Number of output features."},
    };

    message<> list { this, "list", "Input to the network.",
        MIN_FUNCTION {
            lock lock {m_mutex};

            std::vector<float> x = from_atoms<std::vector<float>>(args);
            if (m_output.size() != out_features)
            {
                m_output.resize(out_features);
            }

            auto tensor = Model::getInstance().toTensor(
                x.data(),
                1,
                1,
                1,
                x.size()
            );

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
            cout << "fc - " << VERSION << " - 2022 - Rodrigo Diaz" << endl;
            return {};
        }
    };
private:
    std::mutex m_mutex;
    std::vector<float> m_output;
};

FC::FC(const atoms &args) {

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
        cout << "model with path: " << modelPath << " could not be loaded" << endl;
    }
}

FC::~FC() {}

MIN_EXTERNAL(FC);
