#include "c74_min.h"

#include "model.h"
#include "log.h"

#ifndef VERSION
#define VERSION "0.0.0"
#endif

using namespace c74::min;

class FC : public object<FC> {
public:
    MIN_DESCRIPTION	{"Loads and runs a NN."};
    MIN_TAGS		{"torch"};
    MIN_AUTHOR		{"Rodrigo Diaz"};
    MIN_RELATED		{"print"};

    FC(const atoms &args = {});
    ~FC();

	inlet<>		input	{ this, "(list) flattened input" };
	outlet<>	output	{ this, "(list) features" };
    
    message<> load { this, "load", "Load a model from a file",
        MIN_FUNCTION {
            if (args.size() == 1)
            {
                auto modelPath = std::string(args[0]);
                loadModel(modelPath);
            }
            return {};
        }
    };
    
    attribute<int> out_features
    { 
        this,
        "out_features",
        32 * 2 * 6,
        description {"Number of output features."},
    };

    message<> set { this, "set", "Set a parameter of the model.",
        MIN_FUNCTION {
            if (args.size() == 2)
            {
                auto name = std::string(args[0]);
                auto value = args[1];

                lock lock {m_mutex};
                Model::getInstance().callMethod(name, value);
                lock.unlock();
            }
            return {};
        }
    };
    message<> list { this, "list", "Input to the network.",
        MIN_FUNCTION {
            lock lock {m_mutex};

            if (!m_loaded)
            {
                cout << "Model not loaded." << endl;
                return {};
            }

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
    void loadModel(const std::string &path);

private:
    bool m_loaded {false};
    std::mutex m_mutex;
    std::vector<float> m_output;
};

FC::FC(const atoms &args)
{
    if (!args.empty())
    {
        auto modelPath = std::string(args[0]);
        loadModel(modelPath);
    }
}

void FC::loadModel(const std::string &path)
{
    if (0 == Model::getInstance().loadModel(path, "cpu"))
    {
        cout << "model with path: " << path << " loaded" << endl;
        m_loaded = true;
    }
    else
    {
        cout << "model with path: " << path << " could not be loaded" << endl;
    }
}

FC::~FC() {}

MIN_EXTERNAL(FC);
